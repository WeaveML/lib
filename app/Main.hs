module Main where

import Weave.Base
import Weave.Train
import Weave.Dataset
import Data.Binary (encodeFile)
import System.Random (newStdGen)
import System.IO (hSetBuffering, stdout, BufferMode(NoBuffering))
import Text.Printf (printf)

evaluateAccuracy :: Network -> [(Vector, Int)] -> Double
evaluateAccuracy net testData =
  let correctPredictions = filter (\(img, label) -> argMax (predict net img) == label) testData
      correctCount = length correctPredictions
      totalCount = length testData
  in (fromIntegral correctCount / fromIntegral totalCount) * 100.0

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering

  putStr "Generating datasets... "
  trainDataset <- generateDataset 1500 0.20
  testDataset  <- generateDataset 300 0.20
  putStrLn "[Готово]"
  
  gen <- newStdGen
  let initialNet = initRandomNetwork gen
  putStrLn "[Готово]"
  
  putStrLn "\nStart learning:"
  putStrLn "--------------------------------------------"
  
  finalNet <- foldlMSteps initialNet [1..20] trainDataset testDataset
  
  putStrLn "--------------------------------------------"
  putStrLn "Learning finished!"
  
  let modelPath = "weave_0.1.0.0.bin"
  putStr $ "Saving into: " ++ modelPath ++ "... "
  encodeFile modelPath finalNet
  putStrLn "[Success]"

foldlMSteps :: Network -> [Int] -> [(Vector, Int)] -> [(Vector, Int)] -> IO Network
foldlMSteps net [] _ _ = return net
foldlMSteps net (epoch:epochs) trainData testData = do
  let trainedNet = trainEpoch net trainData
  
  let trainAcc = evaluateAccuracy trainedNet trainData
  let testAcc  = evaluateAccuracy trainedNet testData
  
  printf "Epoch %02d/20 | Global accuracy: %6.2f%% | Accuracy per test: %6.2f%%\n" 
         epoch trainAcc testAcc
         
  foldlMSteps trainedNet epochs trainData testData
