module Main (main) where

import Weave.Base (Network(..), predict, argMax)

dummyNet :: Network 
dummyNet = Network 
  { wHidden = replicate 64 $ replicate 784 0.01
  , bHidden = replicate 64 0.1
  , wOutput = replicate 10 $ replicate 64 0.02
  , bOutput = replicate 10 0.1
  }

main :: IO ()
main = 
  let dummyImage = replicate 784 0.5
      probabilities = predict dummyNet dummyImage
      guess = argMax probabilities
  in do 
    print probabilities
    putStrLn $ "Guess: " ++ show guess
