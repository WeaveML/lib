module Main (main) where

import qualified Data.Vector.Unboxed as U
import           Data.Vector.Unboxed ((!))

import Weave.Base
import Weave.Train
import Weave.Dataset
import Weave.IO
import CLI

import System.Random (newStdGen)
import System.IO     (hSetBuffering, stdout, BufferMode(NoBuffering))
import qualified Data.ByteString as B
import Text.Printf   (printf)

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  cmd <- parseCLI

  case cmd of
    Train numEpochs outputPath -> do
      putStrLn "Generating datasets... "
      trainDataset <- generateDataset 1500 0.3
      testDataset  <- generateDataset 300  0.15
      putStrLn "done."

      gen <- newStdGen
      let initialNet = initRandomNetwork gen

      putStrLn "Training..."
      finalNet <- foldlMSteps initialNet [1..numEpochs] numEpochs trainDataset testDataset

      printf "\nSaving model: %s... " outputPath
      saveModel finalNet outputPath
      putStrLn "done."

    Predict modelPath imgPath -> do
      putStrLn $ "Loading model: " ++ modelPath ++ "... "
      net <- loadModel modelPath :: IO Network
      putStrLn "done."

      rawBytes <- B.readFile imgPath
      let expected = imgH * imgW
      if B.length rawBytes /= expected
        then error $ printf "Expected %d bytes (%dx%d), got %d."
                            expected imgH imgW (B.length rawBytes)
        else do
          let inputVec   = map (\b -> fromIntegral b / 255.0) (B.unpack rawBytes)
              probs      = predict net inputVec
              guess      = argMax probs

          printf "\nPredicted digit: %d\n\n" guess
          putStrLn "Class probabilities:"
          mapM_ (\i -> printf "  %d : %6.2f%%\n" i (probs ! i * 100))
                ([0..9] :: [Int])

foldlMSteps :: Network -> [Int] -> Int
            -> [(Vector, Int)] -> [(Vector, Int)]
            -> IO Network
foldlMSteps net [] _ _ _ = return net
foldlMSteps net (epoch:epochs) total trainData testData = do
  let trained   = trainEpoch net trainData
      trainAcc  = evaluateAccuracy trained trainData
      testAcc   = evaluateAccuracy trained testData
  printf "Epoch %02d/%02d | Train: %6.2f%% | Test: %6.2f%%\n"
         epoch total trainAcc testAcc
  foldlMSteps trained epochs total trainData testData

evaluateAccuracy :: Network -> [(Vector, Int)] -> Double
evaluateAccuracy net dataset =
  let correct = length $ filter (\(img, lbl) -> argMax (predict net img) == lbl) dataset
  in fromIntegral correct / fromIntegral (length dataset) * 100.0
