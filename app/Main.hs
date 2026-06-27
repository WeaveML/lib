module Main where

import Weave.Base 
import Weave.Train
import Weave.Dataset
import Weave.IO
import CLI

import System.Random (newStdGen)
import System.IO (hSetBuffering, stdout, BufferMode(NoBuffering))
import qualified Data.ByteString as B
import Text.Printf (printf)

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  cmd <- parseCLI
  
  case cmd of
    Train numEpochs outputPath -> do
      putStr "Generation datasets... "
      trainDataset <- generateDataset 150 0.4
      testDataset  <- generateDataset 30 0.25
      
      gen <- newStdGen
      let initialNet = initRandomNetwork gen
      
      putStr "Training... "
      finalNet <- foldlMSteps initialNet [1..numEpochs] numEpochs trainDataset testDataset
      
      printf "\nSaving model: %s... " outputPath
      saveModel finalNet outputPath 

    Predict modelPath imgPath -> do
      putStr $ "Model: " ++ modelPath ++ "... "
      net <- loadModel modelPath :: IO Network
      
      rawBytes <- B.readFile imgPath
      if B.length rawBytes /= 10000
        then error $ printf "Error: expected (100x100) 10kb, found %db file." (B.length rawBytes)
        else do
          let inputVector = map (\b -> fromIntegral b / 255.0) (B.unpack rawBytes)
          
          let probabilities = predict net inputVector
          let guess = argMax probabilities
          
          printf "\nAnalyse total:\n"
          printf "Network says, answer is: %d\n\n" guess
          putStrLn "Maybes:"
          mapM_ (\(i, p) -> printf "Number %d: %6.2f%%\n" i (p * 100)) (zip ([0..9] :: [Int]) probabilities)

foldlMSteps :: Network -> [Int] -> Int -> [(Vector, Int)] -> [(Vector, Int)] -> IO Network
foldlMSteps net [] _ _ _ = return net
foldlMSteps net (epoch:epochs) total trainData testData = do
  let trainedNet = trainEpoch net trainData
  let trainAcc = evaluateAccuracy trainedNet trainData
  let testAcc  = evaluateAccuracy trainedNet testData
  printf "Epoch %02d/%02d | Global accuracy: %6.2f%% | Accuracy per test: %6.2f%%\n" 
         epoch total trainAcc testAcc
  foldlMSteps trainedNet epochs total trainData testData

evaluateAccuracy :: Network -> [(Vector, Int)] -> Double
evaluateAccuracy net testData =
  let correct = filter (\(img, label) -> argMax (predict net img) == label) testData
  in (fromIntegral (length correct) / fromIntegral (length testData)) * 100.0
