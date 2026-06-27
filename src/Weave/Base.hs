{-# LANGUAGE DeriveGeneric #-}

module Weave.Base
  ( Network(..)
  , ConvFilter
  , FeatureMap
  , argMax
  , predict
  , relu
  , softMax
  , layerForward
  , dot
  , Matrix
  , Vector
  , Image
  , initRandomNetwork
  , chunksOf
  , conv2dSingle
  , conv2d
  , maxPool2x2
  , flatten
  , applyFilters
  , imgW, imgH
  , after_pool1_h, after_pool1_w
  , after_conv1_h, after_conv1_w
  , after_conv2_h, after_conv2_w
  , after_pool2_h, after_pool2_w
  , conv1Filters, conv1KernelSize
  , conv2Filters, conv2KernelSize
  , fc1Size, numClasses, flatSize
  ) where

import System.Random (StdGen, randomRs, split)
import Data.Binary (Binary)
import GHC.Generics (Generic)

type Vector     = [Double]
type Matrix     = [Vector]
type Image      = (Vector, Int)
type FeatureMap = [[Double]]
type ConvFilter = [[Double]]

imgW, imgH :: Int
imgW = 100
imgH = 100

conv1Filters, conv1KernelSize :: Int
conv1Filters    = 8
conv1KernelSize = 5

conv2Filters, conv2KernelSize :: Int
conv2Filters    = 16
conv2KernelSize = 3

fc1Size, numClasses :: Int
fc1Size    = 128
numClasses = 10

after_conv1_h, after_conv1_w :: Int
after_conv1_h = imgH - conv1KernelSize + 1  
after_conv1_w = imgW - conv1KernelSize + 1  

after_pool1_h, after_pool1_w :: Int
after_pool1_h = after_conv1_h `div` 2  
after_pool1_w = after_conv1_w `div` 2  -- 48

after_conv2_h, after_conv2_w :: Int
after_conv2_h = after_pool1_h - conv2KernelSize + 1  
after_conv2_w = after_pool1_w - conv2KernelSize + 1  -- 46

after_pool2_h, after_pool2_w :: Int
after_pool2_h = after_conv2_h `div` 2  
after_pool2_w = after_conv2_w `div` 2  -- 23

flatSize :: Int
flatSize = after_pool2_h * after_pool2_w * conv2Filters  -- 8464

data Network = Network
  { convW1  :: [ConvFilter]    
  , convB1  :: Vector         
  , convW2  :: [[ConvFilter]]
  , convB2  :: Vector       
  , wHidden :: Matrix      
  , bHidden :: Vector     
  , wOutput :: Matrix          
  , bOutput :: Vector         
  } deriving (Show, Generic)

instance Binary Network

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

initRandomNetwork :: StdGen -> Network
initRandomNetwork gen =
  let (g1,r1) = split gen
      (g2,r2) = split r1
      (g3,r3) = split r2
      (g4,_)  = split r3

      -- хавьер бля: limit = sqrt(6 / (fan_in + fan_out))
      scale1 = sqrt (6.0 / fromIntegral (conv1KernelSize * conv1KernelSize + conv1Filters))
      scale2 = sqrt (6.0 / fromIntegral (conv2KernelSize * conv2KernelSize * conv1Filters + conv2Filters))
      scale3 = sqrt (6.0 / fromIntegral (flatSize + fc1Size))
      scale4 = sqrt (6.0 / fromIntegral (fc1Size + numClasses))

      rw1 = randomRs (-scale1, scale1) g1
      rw2 = randomRs (-scale2, scale2) g2
      rw3 = randomRs (-scale3, scale3) g3
      rw4 = randomRs (-scale4, scale4) g4

      k1sz = conv1KernelSize * conv1KernelSize
      k2sz = conv2KernelSize * conv2KernelSize

      cw1 = [ chunksOf conv1KernelSize
                (take k1sz (drop (i * k1sz) rw1))
            | i <- [0 .. conv1Filters - 1] ]
      cb1 = replicate conv1Filters 0.0

      cw2 = [ [ chunksOf conv2KernelSize
                  (take k2sz (drop ((fi * conv1Filters + ci) * k2sz) rw2))
              | ci <- [0 .. conv1Filters - 1] ]
            | fi <- [0 .. conv2Filters - 1] ]
      cb2 = replicate conv2Filters 0.0

      wH = chunksOf flatSize (take (fc1Size * flatSize) rw3)
      bH = replicate fc1Size 0.0

      -- output
      wO = chunksOf fc1Size (take (numClasses * fc1Size) rw4)
      bO = replicate numClasses 0.0

  in Network cw1 cb1 cw2 cb2 wH bH wO bO

dot :: Vector -> Vector -> Double
dot xs ys = sum (zipWith (*) xs ys)

layerForward :: Matrix -> Vector -> Vector -> Vector
layerForward ws inp b = zipWith (+) (map (`dot` inp) ws) b

relu :: Double -> Double
relu x = max 0.0 x

softMax :: Vector -> Vector
softMax xs =
  let m    = maximum xs
      exps = map (\x -> exp (x - m)) xs
      s    = sum exps
  in map (/ s) exps

argMax :: Vector -> Int
argMax xs =
  fst $ foldl1 (\(im,vm) (i,v) -> if v > vm then (i,v) else (im,vm))
               (zip [0..] xs)

-- свертыч 

conv2dSingle :: ConvFilter -> FeatureMap -> FeatureMap
conv2dSingle kernel fm =
  let kH   = length kernel
      kW   = length (head kernel)
      outH = length fm - kH + 1
      outW = length (head fm) - kW + 1
  in [ [ sumPatch kH kW r c | c <- [0 .. outW - 1] ]
     | r <- [0 .. outH - 1] ]
  where
    sumPatch kH kW r c =
      sum [ (kernel !! kr !! kc) * (fm !! (r + kr) !! (c + kc))
          | kr <- [0 .. kH - 1]
          , kc <- [0 .. kW - 1] ]

conv2d :: [ConvFilter] -> [FeatureMap] -> Double -> FeatureMap
conv2d kernels inputs bias =
  let partials = zipWith conv2dSingle kernels inputs
      addFM    = zipWith (zipWith (+))
      summed   = foldl1 addFM partials
  in map (map (+ bias)) summed

applyFilters :: [[ConvFilter]] -> Vector -> [FeatureMap] -> [FeatureMap]
applyFilters convWs biases inputs =
  zipWith (\ks b -> conv2d ks inputs b) convWs biases

maxPool2x2 :: FeatureMap -> FeatureMap
maxPool2x2 fm =
  let outH = length fm `div` 2
      outW = length (head fm) `div` 2
  in [ [ maximum [ fm !! (r*2 + dr) !! (c*2 + dc)
                 | dr <- [0,1], dc <- [0,1] ]
       | c <- [0 .. outW - 1] ]
     | r <- [0 .. outH - 1] ]

flatten :: [FeatureMap] -> Vector
flatten = concatMap concat

predict :: Network -> Vector -> Vector
predict net input =
  let inputFM   = chunksOf imgW input

      conv1Raw  = applyFilters (map (: []) (convW1 net)) (convB1 net) [inputFM]
      relu1     = map (map (map relu)) conv1Raw
      pool1     = map maxPool2x2 relu1          

      conv2Raw  = applyFilters (convW2 net) (convB2 net) pool1
      relu2     = map (map (map relu)) conv2Raw
      pool2     = map maxPool2x2 relu2          -- 23×23×16

      flatVec    = flatten pool2                 -- 8464
      hiddenRaw  = layerForward (wHidden net) flatVec (bHidden net)
      hiddenAct  = map relu hiddenRaw
      outputRaw  = layerForward (wOutput net) hiddenAct (bOutput net)
  in softMax outputRaw
