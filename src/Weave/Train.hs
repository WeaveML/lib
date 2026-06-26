module Weave.Train (trainStep, reluDerivative, trainEpoch) where

import Weave.Base ( Network(..)
                  , Vector
                  , Matrix
                  , dot
                  , relu
                  , layerForward
                  , softMax
                  , Image
                  )
import Data.List (transpose)

learningRate :: Double
learningRate = 0.01 

reluDerivative :: Double -> Double 
reluDerivative x = if x > 0 then 1.0 else 0.0

trainStep :: Network -> Vector -> Int -> Network 
trainStep net input targetIdx = 
  let 
    hiddenRaw = layerForward (wHidden net) input (bHidden net)
    hiddenAct = map relu hiddenRaw 

    outputRaw = layerForward (wOutput net) hiddenAct (bOutput net)
    outputAct = softMax outputRaw

    targetVector = [if i == targetIdx then 1.0 else 0.0 | i <- [0..9]]

    outputDelta = zipWith (-) outputAct targetVector

    wOutputT = transpose (wOutput net)
    hiddenDeltaRaw = map (`dot` outputDelta) wOutputT

    hiddenDelta = zipWith (*) hiddenDeltaRaw (map reluDerivative hiddenRaw)

    dwOutput = [[d * x | x <- hiddenAct] | d <- outputDelta]
    dwHidden = [[d * x | x <- input] | d <- hiddenDelta]
    
    updateParam :: Double -> Double -> Double
    updateParam old grad = old - (learningRate * grad)
      
    updateMatrix :: Matrix -> Matrix -> Matrix
    updateMatrix = zipWith (zipWith updateParam)
      
    updateVector :: Vector -> Vector -> Vector
    updateVector = zipWith updateParam


  in Network
      { wHidden = updateMatrix (wHidden net) dwHidden
      , bHidden = updateVector (bHidden net) hiddenDelta 
      , wOutput = updateMatrix (wOutput net) dwOutput
      , bOutput = updateVector (bOutput net) outputDelta
      }


trainEpoch :: Network -> [Image] -> Network
trainEpoch net dataset = foldl (\currentNet (img, label) -> trainStep currentNet img label) net dataset 
