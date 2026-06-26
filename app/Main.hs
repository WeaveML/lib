module Main (main) where

import Weave.Base 
import Weave.Train 
import Weave.Dataset 
import Control.Monad (forM_)

dummyNet :: Network 
dummyNet = Network 
  { wHidden = replicate 64 $ replicate 784 0.01
  , bHidden = replicate 64 0.1
  , wOutput = replicate 10 $ replicate 64 0.02
  , bOutput = replicate 10 0.1
  }

main :: IO ()
main = do 
  dataset <- generateDataset 1000 0.25
  
  putStrLn "Learning (5 epoch)..."
  let trainedNet = iterate (`trainEpoch` dataset) dummyNet  !! 5
  
  (testThree, _) <- generateSample 3 0.20
  
  let probabilities = predict trainedNet testThree
  let guess = argMax probabilities
  
  putStrLn "Tests (0..9):"
  forM_ (zip [0..9] probabilities) $ \(digit, prob) ->
    putStrLn $ "Number " ++ show digit ++ ": " ++ show (prob * 100) ++ "%"
