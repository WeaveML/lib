{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BangPatterns  #-}

module Weave.Base
  ( Network(..)
  , UVec
  , UMat         
  , FMap          
  , predict
  , argMax
  , relu, softMax
  , layerForward
  , dot
  , flatten
  , conv2dSingle
  , applyFilters
  , maxPool2x2
  , initRandomNetwork
  , chunksToFMap, fmapAdd
  , imgW, imgH
  , c1F, k1, c2F, k2
  , fc1, nCls, flatSz
  , outH1, outW1, pH1, pW1
  , outH2, outW2, pH2, pW2
  , Image, Vector
  ) where

import qualified Data.Vector.Unboxed         as U
import           Data.Vector.Unboxed         ((!))
import           System.Random               (StdGen, randomRs, split)
import           Data.Binary                 (Binary(..))
import           GHC.Generics                (Generic)

type UVec  = U.Vector Double
type UMat  = (Int, Int, UVec)   
type FMap  = (Int, Int, UVec)  

type Vector = [Double]
type Image  = (Vector, Int)

imgW, imgH :: Int
imgW = 100 ; imgH = 100

c1F, k1 :: Int   
c1F = 8  ; k1 = 5

c2F, k2 :: Int  
c2F = 16 ; k2 = 3

fc1, nCls :: Int
fc1 = 128 ; nCls = 10

outH1, outW1 :: Int          
outH1 = imgH - k1 + 1       
outW1 = imgW - k1 + 1      

pH1, pW1 :: Int           
pH1 = outH1 `div` 2      
pW1 = outW1 `div` 2       

outH2, outW2 :: Int      
outH2 = pH1 - k2 + 1    
outW2 = pW1 - k2 + 1   

pH2, pW2 :: Int       
pH2 = outH2 `div` 2  
pW2 = outW2 `div` 2 

flatSz :: Int
flatSz = pH2 * pW2 * c2F    

data Network = Network
  { nConvW1  :: U.Vector Double  
  , nConvB1  :: U.Vector Double 
  , nConvW2  :: U.Vector Double  
  , nConvB2  :: U.Vector Double 
  , nWHid    :: U.Vector Double  
  , nBHid    :: U.Vector Double 
  , nWOut    :: U.Vector Double
  , nBOut    :: U.Vector Double  
  } deriving (Show, Generic)

instance Binary Network where
  put n = do
    put (U.toList (nConvW1 n)); put (U.toList (nConvB1 n))
    put (U.toList (nConvW2 n)); put (U.toList (nConvB2 n))
    put (U.toList (nWHid   n)); put (U.toList (nBHid   n))
    put (U.toList (nWOut   n)); put (U.toList (nBOut   n))
  get = do
    cw1 <- U.fromList <$> get; cb1 <- U.fromList <$> get
    cw2 <- U.fromList <$> get; cb2 <- U.fromList <$> get
    wh  <- U.fromList <$> get; bh  <- U.fromList <$> get
    wo  <- U.fromList <$> get; bo  <- U.fromList <$> get
    return (Network cw1 cb1 cw2 cb2 wh bh wo bo)


initRandomNetwork :: StdGen -> Network
initRandomNetwork gen =
  let (g1,r1) = split gen; (g2,r2) = split r1
      (g3,r3) = split r2;  (g4,_)  = split r3

      xav n m = sqrt (6.0 / fromIntegral (n + m))

      s1 = xav (k1*k1)        c1F
      s2 = xav (k2*k2*c1F)    c2F
      s3 = xav flatSz          fc1
      s4 = xav fc1             nCls

      take' n s g = U.fromList . take n $ randomRs (-s, s) g

  in Network
       { nConvW1 = take' (c1F * k1*k1)       s1 g1
       , nConvB1 = U.replicate c1F 0.0
       , nConvW2 = take' (c2F * c1F * k2*k2) s2 g2
       , nConvB2 = U.replicate c2F 0.0
       , nWHid   = take' (fc1 * flatSz)       s3 g3
       , nBHid   = U.replicate fc1 0.0
       , nWOut   = take' (nCls * fc1)         s4 g4
       , nBOut   = U.replicate nCls 0.0
       }

relu :: Double -> Double
relu x = if x > 0 then x else 0

softMax :: UVec -> UVec
softMax v =
  let m = U.maximum v
      e = U.map (\x -> exp (x - m)) v
      s = U.sum e
  in U.map (/ s) e

dot :: UVec -> UVec -> Double
dot a b = U.sum (U.zipWith (*) a b)

layerForward :: UMat -> UVec -> UVec -> UVec
layerForward (rows, cols, ws) inp bias =
  U.generate rows $ \i ->
    let row = U.slice (i * cols) cols ws
    in dot row inp + bias ! i

argMax :: UVec -> Int
argMax v = U.maxIndex v

fmGet :: FMap -> Int -> Int -> Double
fmGet (_, w, d) r c = d ! (r * w + c)
{-# INLINE fmGet #-}

mkFMap :: Int -> Int -> UVec -> FMap
mkFMap h w v = (h, w, v)

fmapAdd :: FMap -> FMap -> FMap
fmapAdd (h,w,a) (_,_,b) = (h, w, U.zipWith (+) a b)

zeroFMap :: Int -> Int -> FMap
zeroFMap h w = (h, w, U.replicate (h*w) 0.0)

chunksToFMap :: Int -> Int -> UVec -> FMap
chunksToFMap h w v = (h, w, v)

conv2dSingle :: Int -> Int  
             -> UVec       
             -> FMap      
             -> FMap     
conv2dSingle kH kW kernel (inH, inW, inp) =
  let !outH = inH - kH + 1
      !outW = inW - kW + 1
      !n    = outH * outW
      v = U.generate n $ \idx ->
            let !r  = idx `div` outW
                !c  = idx `mod` outW
            in U.sum $ U.generate (kH * kW) $ \kid ->
                 let !kr = kid `div` kW
                     !kc = kid `mod` kW
                 in (kernel ! kid) * (inp ! ((r+kr)*inW + (c+kc)))
  in (outH, outW, v)

conv2dMulti :: Int -> Int   
            -> [UVec]      
            -> [FMap]     
            -> Double    
            -> FMap
conv2dMulti kH kW kernels inputs bias =
  let partials = zipWith (conv2dSingle kH kW) kernels inputs
      (h, w, _) = head partials
      summed = foldl1 (\(_,_,a) (_,_,b) -> (h, w, U.zipWith (+) a b)) partials
      (_, _, sv) = summed
  in (h, w, U.map (+ bias) sv)

applyConv1 :: Network -> FMap -> [FMap]
applyConv1 net inputFM =
  [ let !off    = fi * k1*k1
        !kernel = U.slice off (k1*k1) (nConvW1 net)
        !bias   = nConvB1 net ! fi
        (h, w, sv) = conv2dSingle k1 k1 kernel inputFM
    in (h, w, U.map (+ bias) sv)
  | fi <- [0 .. c1F - 1] ]

applyConv2 :: Network -> [FMap] -> [FMap]
applyConv2 net pool1 =
  [ let kernels = [ U.slice ((fi*c1F + ci) * k2*k2) (k2*k2) (nConvW2 net)
                  | ci <- [0..c1F-1] ]
        !bias   = nConvB2 net ! fi
    in conv2dMulti k2 k2 kernels pool1 bias
  | fi <- [0 .. c2F - 1] ]

applyFilters :: Network -> [FMap] -> [FMap]
applyFilters = applyConv2


maxPool2x2 :: FMap -> FMap
maxPool2x2 (inH, inW, inp) =
  let !outH = inH `div` 2
      !outW = inW `div` 2
      v = U.generate (outH * outW) $ \idx ->
            let !r = idx `div` outW
                !c = idx `mod` outW
            in max (max (inp ! ((r*2)  *inW + c*2))
                        (inp ! ((r*2)  *inW + c*2+1)))
                   (max (inp ! ((r*2+1)*inW + c*2))
                        (inp ! ((r*2+1)*inW + c*2+1)))
  in (outH, outW, v)

flatten :: [FMap] -> UVec
flatten fmaps = U.concat [ v | (_,_,v) <- fmaps ]

predict :: Network -> Vector -> UVec
predict net inputList =
  let inputVec = U.fromList inputList
      inputFM  = (imgH, imgW, inputVec)

      conv1    = applyConv1 net inputFM
      relu1    = map (\(h,w,v) -> (h,w, U.map relu v)) conv1
      pool1    = map maxPool2x2 relu1

      conv2    = applyConv2 net pool1
      relu2    = map (\(h,w,v) -> (h,w, U.map relu v)) conv2
      pool2    = map maxPool2x2 relu2

      flatVec  = flatten pool2
      hidRaw   = layerForward (fc1, flatSz, nWHid net) flatVec (nBHid net)
      hidAct   = U.map relu hidRaw
      outRaw   = layerForward (nCls, fc1, nWOut net) hidAct (nBOut net)
  in softMax outRaw
