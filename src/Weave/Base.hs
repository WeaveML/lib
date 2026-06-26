{-# LANGUAGE DeriveGeneric #-}

module Weave.Base 
  ( Network(..)
  , argMax
  , predict
  , layerForward
  , relu 
  , softMax 
  , Matrix
  , Vector 
  , Image 
  , dot
  , initRandomNetwork
  ) where

import System.Random (StdGen, randomRs, split)
import Data.Binary (Binary)
import GHC.Generics (Generic)

type Vector = [Double]
type Matrix = [Vector]
type Image = (Vector, Int) 

data Network = Network 
  { wHidden :: Matrix
  , bHidden :: Vector 
  , wOutput :: Matrix 
  , bOutput :: Vector
  } deriving (Show, Generic)

instance Binary Network

initRandomNetwork :: StdGen -> Network
initRandomNetwork gen =
  let (g1, tmp1) = split gen
      (g2, tmp2) = split tmp1
      (g3, g4)   = split tmp2
      
      randW1 = randomRs (-0.05, 0.05) g1
      randW2 = randomRs (-0.05, 0.05) g2
      
      chunksOf :: Int -> [a] -> [[a]]
      chunksOf _ [] = []
      chunksOf n xs = take n xs : chunksOf n (drop n xs)
      
      wH = chunksOf 784 (take (64 * 784) randW1)
      bH = replicate 64 0.0
      
      wO = chunksOf 64 (take (10 * 64) randW2)
      bO = replicate 10 0.0
  in Network wH bH wO bO

-- scalar * vectors
dot :: Vector -> Vector -> Double 
dot xs ys = sum $ zipWith (*) xs ys

layerForward :: Matrix -> Vector -> Vector -> Vector 
layerForward weightMatrix inputVector bias =
  zipWith (+) dotHelp bias 
  where 
    dotHelp = map (`dot` inputVector) weightMatrix

relu :: Double -> Double 
relu x = max 0 x

softMax :: Vector -> Vector 
softMax xs =
  let exps = map exp xs
      sumExps = sum exps
  in map (/ sumExps) exps

predict :: Network -> Vector -> Vector 
predict net input = 
  let 
    hiddenLayerAct = map relu $ layerForward (wHidden net) input (bHidden net)

    outputLayerRaw = layerForward (wOutput net) hiddenLayerAct (bOutput net)
  in 
    softMax outputLayerRaw

argMax :: Vector -> Int
argMax xs = fst $ foldl1 (\(idxMax, valMax) (idx, val) -> 
  if val > valMax then (idx, val) else (idxMax, valMax)) (zip [0..] xs)
