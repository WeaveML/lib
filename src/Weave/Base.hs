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
  ) where

type Vector = [Double]
type Matrix = [Vector]
type Image = (Vector, Int) 

data Network = Network 
  { wHidden :: Matrix
  , bHidden :: Vector 
  , wOutput :: Matrix 
  , bOutput :: Vector
  } deriving Show

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
