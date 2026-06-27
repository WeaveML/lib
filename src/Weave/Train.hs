{-# LANGUAGE BangPatterns #-}

module Weave.Train (trainStep, trainEpoch, reluDerivative) where

import qualified Data.Vector.Unboxed         as U
import qualified Data.Vector.Unboxed.Mutable as MU
import           Data.Vector.Unboxed         ((!))
import           Control.Monad               (forM_)
import           Control.Monad.ST            (runST)

import Weave.Base

lr :: Double
lr = 1e-4

reluDerivative :: Double -> Double
reluDerivative x = if x > 0 then 1.0 else 0.0

applyGrad :: UVec -> UVec -> UVec
applyGrad params grads = U.zipWith (\p g -> p - lr * g) params grads

maxPool2x2Fwd :: FMap -> (FMap, U.Vector Int)
maxPool2x2Fwd (inH, inW, inp) =
  let !outH = inH `div` 2
      !outW = inW `div` 2
      !n    = outH * outW
      (vals, idxs) = U.unzip $ U.generate n $ \i ->
        let !r  = i `div` outW
            !c  = i `mod` outW
            candidates = [ (inp ! ((r*2+dr)*inW + c*2+dc), (r*2+dr)*inW + c*2+dc)
                         | dr <- [0,1], dc <- [0,1] ]
            (v, idx) = foldl1 (\a b -> if fst a >= fst b then a else b) candidates
        in (v, idx)
  in ((outH, outW, vals), idxs)

maxPoolBwd :: U.Vector Int  
           -> FMap         
           -> Int -> Int  
           -> FMap
maxPoolBwd mask (outH, outW, dOut) inH inW = runST $ do
  mv <- MU.replicate (inH * inW) 0.0
  forM_ [0 .. outH*outW - 1] $ \i -> do
    let !g   = dOut ! i
        !idx = mask ! i
    old <- MU.read mv idx
    MU.write mv idx (old + g)
  v <- U.freeze mv
  return (inH, inW, v)

gradKernel :: Int -> Int 
           -> FMap        
           -> FMap      
           -> UVec     
gradKernel kH kW (_, inW, inp) (outH, outW, dOut) =
  U.generate (kH * kW) $ \kid ->
    let !kr = kid `div` kW
        !kc = kid `mod` kW
    in U.sum $ U.generate (outH * outW) $ \i ->
         let !r = i `div` outW
             !c = i `mod` outW
         in (dOut ! i) * (inp ! ((r+kr)*inW + (c+kc)))

gradInput :: Int -> Int  
          -> UVec       
          -> FMap      
          -> FMap     
gradInput kH kW kernel (outH, outW, dOut) =
  let !pH  = kH - 1
      !pW  = kW - 1
      !_ = outH + 2*pH
      !_ = outW + 2*pW
      flipped = U.reverse kernel
      !inH = outH + kH - 1
      !inW = outW + kW - 1
      v = U.generate (inH * inW) $ \idx ->
            let !r = idx `div` inW
                !c = idx `mod` inW
            in U.sum $ U.generate (kH * kW) $ \kid ->
                 let !kr  = kid `div` kW
                     !kc  = kid `mod` kW
                     !pr  = r + pH - kr  
                     !pc  = c + pW - kc
                 in if pr >= 0 && pr < outH && pc >= 0 && pc < outW
                    then (flipped ! kid) * (dOut ! (pr*outW + pc))
                    else 0.0
  in (inH, inW, v)

trainStep :: Network -> Vector -> Int -> Network
trainStep net inputList targetIdx =
  let
    inputVec = U.fromList inputList
    inputFM  = (imgH, imgW, inputVec)

    conv1Raw = applyConv1' net inputFM        
    relu1    = mapFM relu conv1Raw
    (pool1, mask1) = unzip (map maxPool2x2Fwd relu1)  

    conv2Raw = applyConv2' net pool1          
    relu2    = mapFM relu conv2Raw
    (pool2, mask2) = unzip (map maxPool2x2Fwd relu2) 

    flatVec  = flatten pool2                        
    hidRaw   = layerForward (fc1, flatSz, nWHid net) flatVec (nBHid net)
    hidAct   = U.map relu hidRaw
    outRaw   = layerForward (nCls, fc1, nWOut net) hidAct (nBOut net)
    outAct   = softMax outRaw


    target = U.generate nCls (\i -> if i == targetIdx then 1.0 else 0.0)
    dOut   = U.zipWith (-) outAct target              

    dwOut = outerProd dOut hidAct                      
    dbOut = dOut

    dHidAct = matVecT (nCls, fc1, nWOut net) dOut       
    dHidRaw = U.zipWith (*) dHidAct (U.map reluDerivative hidRaw)

    dwHid = outerProd dHidRaw flatVec                   
    dbHid = dHidRaw

    dFlatVec = matVecT (fc1, flatSz, nWHid net) dHidRaw 

    dPool2 = splitFMaps c2F pH2 pW2 dFlatVec

    dRelu2 = zipWith (\dP mk -> maxPoolBwd mk dP outH2 outW2) dPool2 mask2

    dConv2 = zipWith (\(h,w,dr) (_, _, rc) ->
                        (h, w, U.zipWith (*) dr (U.map reluDerivative rc)))
                     dRelu2 conv2Raw

    dwConv2 = U.concat
      [ U.concat
        [ gradKernel k2 k2 (pool1 !! ci) (dConv2 !! fi)
        | ci <- [0..c1F-1] ]
      | fi <- [0..c2F-1] ]
    dbConv2 = U.fromList [ let (_,_,v) = dConv2 !! fi in U.sum v | fi <- [0..c2F-1] ]

    dPool1 = [ foldl1 fmapAdd
                 [ gradInput k2 k2
                     (sliceKernel2 fi ci) (dConv2 !! fi)
                 | fi <- [0..c2F-1] ]
             | ci <- [0..c1F-1] ]
      where sliceKernel2 fi ci =
              U.slice ((fi*c1F + ci)*k2*k2) (k2*k2) (nConvW2 net)

    dRelu1 = zipWith (\dP mk -> maxPoolBwd mk dP outH1 outW1) dPool1 mask1

    dConv1 = zipWith (\(h,w,dr) (_,_, rc) ->
                        (h, w, U.zipWith (*) dr (U.map reluDerivative rc)))
                     dRelu1 conv1Raw

    dwConv1 = U.concat
      [ gradKernel k1 k1 inputFM (dConv1 !! fi)
      | fi <- [0..c1F-1] ]
    dbConv1 = U.fromList [ let (_,_,v) = dConv1 !! fi in U.sum v | fi <- [0..c1F-1] ]


  in Network
       { nConvW1 = applyGrad (nConvW1 net) dwConv1
       , nConvB1 = applyGrad (nConvB1 net) dbConv1
       , nConvW2 = applyGrad (nConvW2 net) dwConv2
       , nConvB2 = applyGrad (nConvB2 net) dbConv2
       , nWHid   = applyGrad (nWHid   net) dwHid
       , nBHid   = applyGrad (nBHid   net) dbHid
       , nWOut   = applyGrad (nWOut   net) dwOut
       , nBOut   = applyGrad (nBOut   net) dbOut
       }

mapFM :: (Double -> Double) -> [FMap] -> [FMap]
mapFM f = map (\(h,w,v) -> (h, w, U.map f v))

outerProd :: UVec -> UVec -> UVec
outerProd a b =
  let !na = U.length a
      !nb = U.length b
  in U.generate (na * nb) (\i -> (a ! (i `div` nb)) * (b ! (i `mod` nb)))

matVecT :: UMat -> UVec -> UVec
matVecT (rows, cols, ws) v = runST $ do
  res <- MU.replicate cols 0.0
  forM_ [0..rows-1] $ \i -> do
    let !vi = v ! i
    forM_ [0..cols-1] $ \j -> do
      old <- MU.read res j
      MU.write res j (old + (ws ! (i*cols+j)) * vi)
  U.freeze res

splitFMaps :: Int -> Int -> Int -> UVec -> [FMap]
splitFMaps n h w v =
  [ let !off = i * h * w
    in (h, w, U.slice off (h*w) v)
  | i <- [0..n-1] ]

applyConv1' :: Network -> FMap -> [FMap]
applyConv1' net inputFM =
  [ let !off    = fi * k1*k1
        !kernel = U.slice off (k1*k1) (nConvW1 net)
        !bias   = nConvB1 net ! fi
        (h, w, sv) = conv2dSingle k1 k1 kernel inputFM
    in (h, w, U.map (+ bias) sv)
  | fi <- [0..c1F-1] ]

applyConv2' :: Network -> [FMap] -> [FMap]
applyConv2' net pool1 =
  [ let kernels = [ U.slice ((fi*c1F + ci)*k2*k2) (k2*k2) (nConvW2 net)
                  | ci <- [0..c1F-1] ]
        !bias   = nConvB2 net ! fi
        partials = [ conv2dSingle k2 k2 kg fm | (kg,fm) <- zip kernels pool1 ]
        (h, w, _) = head partials
        sumv = foldl1 (\(_,_,a) (_,_,b) -> (h,w,U.zipWith (+) a b)) partials
        (_,_,sv) = sumv
    in (h, w, U.map (+ bias) sv)
  | fi <- [0..c2F-1] ]


trainEpoch :: Network -> [Image] -> Network
trainEpoch = foldl' (\net (img, lbl) -> trainStep net img lbl)
