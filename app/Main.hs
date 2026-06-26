module Main (main) where

import Weave.Base 
import Weave.Train 
import Weave.Dataset 
import Control.Monad (forM_)

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
