module Weave.Train (trainStep, trainEpoch, reluDerivative) where

import Weave.Base
import Data.List (transpose, maximumBy)
import Data.Ord  (comparing)

learningRate :: Double
learningRate = 1e-4

reluDerivative :: Double -> Double
reluDerivative x = if x > 0 then 1.0 else 0.0

addFM :: FeatureMap -> FeatureMap -> FeatureMap
addFM = zipWith (zipWith (+))

zeroFM :: Int -> Int -> FeatureMap
zeroFM h w = replicate h (replicate w 0.0)

updateV :: Vector -> Vector -> Vector
updateV = zipWith (\old g -> old - learningRate * g)

updateM :: Matrix -> Matrix -> Matrix
updateM = zipWith (zipWith (\old g -> old - learningRate * g))

updateFM :: FeatureMap -> FeatureMap -> FeatureMap
updateFM = zipWith (zipWith (\old g -> old - learningRate * g))


maxPool2x2Fwd :: FeatureMap -> (FeatureMap, [((Int,Int),(Int,Int))])
maxPool2x2Fwd fm =
  let outH  = length fm `div` 2
      outW  = length (head fm) `div` 2
      cells = [ let cands = [ (fm !! (r*2+dr) !! (c*2+dc), (r*2+dr, c*2+dc))
                             | dr <- [0,1], dc <- [0,1] ]
                    (val, pos) = maximumBy (comparing fst) cands
                in (val, (pos, (r,c)))
              | r <- [0..outH-1], c <- [0..outW-1] ]
  in (chunksOf outW (map fst cells), map snd cells)

maxPoolBwd :: [((Int,Int),(Int,Int))] -> FeatureMap -> Int -> Int -> FeatureMap
maxPoolBwd masks dOut inH inW =
  foldl step (zeroFM inH inW) masks
  where
    step acc ((ri,ci),(ro,co)) =
      let g   = dOut !! ro !! co
          row = acc !! ri
          row' = take ci row ++ [(row !! ci) + g] ++ drop (ci+1) row
      in take ri acc ++ [row'] ++ drop (ri+1) acc


gradKernel :: FeatureMap -> FeatureMap -> ConvFilter
gradKernel inputFM dOut =
  let outH = length dOut
      outW = length (head dOut)
      kH   = length inputFM - outH + 1
      kW   = length (head inputFM) - outW + 1
  in [ [ sum [ dOut !! r !! c * inputFM !! (r+kr) !! (c+kc)
             | r <- [0..outH-1], c <- [0..outW-1] ]
       | kc <- [0..kW-1] ]
     | kr <- [0..kH-1] ]


gradInput :: ConvFilter -> FeatureMap -> FeatureMap
gradInput kernel dOut =
  let kH      = length kernel
      kW      = length (head kernel)
      oH      = length dOut
      oW      = length (head dOut)
      pH      = kH - 1
      pW      = kW - 1
      padded  = [ [ if r >= pH && r < pH+oH && c >= pW && c < pW+oW
                    then dOut !! (r-pH) !! (c-pW)
                    else 0.0
                  | c <- [0..oW+2*pW-1] ]
                | r <- [0..oH+2*pH-1] ]
      flipped = map reverse (reverse kernel)
  in conv2dSingle flipped padded

trainStep :: Network -> Vector -> Int -> Network
trainStep net input targetIdx =
  let

    inputFM  = chunksOf imgW input                              

    conv1Raw = applyFilters (map (:[]) (convW1 net)) (convB1 net) [inputFM]
    relu1    = map (map (map relu)) conv1Raw
    (pool1, mask1) = unzip (map maxPool2x2Fwd relu1)

    conv2Raw = applyFilters (convW2 net) (convB2 net) pool1
    relu2    = map (map (map relu)) conv2Raw
    (pool2, mask2) = unzip (map maxPool2x2Fwd relu2)

    -- FC
    flatVec   = flatten pool2                                   -- 8464
    hidRaw    = layerForward (wHidden net) flatVec (bHidden net)
    hidAct    = map relu hidRaw
    outRaw    = layerForward (wOutput net) hidAct (bOutput net)
    outAct    = softMax outRaw

    -- backprog 

    target  = [if i == targetIdx then 1.0 else 0.0 | i <- [0..numClasses-1]]
    dOut    = zipWith (-) outAct target                         -- [10]

    dwOut = [[d * h | h <- hidAct] | d <- dOut]
    dbOut = dOut

    -- - hidAct -> hidRaw
    dHidAct = map (`dot` dOut) (transpose (wOutput net))
    dHidRaw = zipWith (*) dHidAct (map reluDerivative hidRaw)

    -- FC1
    dwHid = [[d * f | f <- flatVec] | d <- dHidRaw]
    dbHid = dHidRaw

    dFlatCh  = chunksOf (after_pool2_h * after_pool2_w)
                 (map (`dot` dHidRaw) (transpose (wHidden net)))
    dPool2FM = map (chunksOf after_pool2_w) dFlatCh             

    dRelu2 = zipWith (\dP mk -> maxPoolBwd mk dP after_conv2_h after_conv2_w)
                     dPool2FM mask2

    dConv2 = zipWith (\dR rC -> zipWith (zipWith (*)) dR (map (map reluDerivative) rC))
                     dRelu2 conv2Raw

    dwConv2 = [ [ gradKernel (pool1 !! ci) (dConv2 !! fi)
                | ci <- [0..conv1Filters-1] ]
              | fi <- [0..conv2Filters-1] ]
    dbConv2 = map (\fm -> sum (map sum fm)) dConv2

    dPool1FM = [ foldl addFM (zeroFM after_pool1_h after_pool1_w)
                   [ gradInput (convW2 net !! fi !! ci) (dConv2 !! fi)
                   | fi <- [0..conv2Filters-1] ]
               | ci <- [0..conv1Filters-1] ]

    dRelu1 = zipWith (\dP mk -> maxPoolBwd mk dP after_conv1_h after_conv1_w)
                     dPool1FM mask1

    dConv1 = zipWith (\dR rC -> zipWith (zipWith (*)) dR (map (map reluDerivative) rC))
                     dRelu1 conv1Raw

    dwConv1 = [ gradKernel inputFM (dConv1 !! fi)
              | fi <- [0..conv1Filters-1] ]
    dbConv1 = map (\fm -> sum (map sum fm)) dConv1


  in Network
       { convW1  = zipWith updateFM (convW1 net) dwConv1
       , convB1  = updateV  (convB1 net) dbConv1
       , convW2  = zipWith (zipWith updateFM) (convW2 net) dwConv2
       , convB2  = updateV  (convB2 net) dbConv2
       , wHidden = updateM  (wHidden net) dwHid
       , bHidden = updateV  (bHidden net) dbHid
       , wOutput = updateM  (wOutput net) dwOut
       , bOutput = updateV  (bOutput net) dbOut
       }


trainEpoch :: Network -> [Image] -> Network
trainEpoch = foldl (\net (img, lbl) -> trainStep net img lbl)
