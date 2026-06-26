module Weave.Dataset 
  ( generateDataset
  , generateSample
  )
  where

import Weave.Base (Vector, Image)
import System.Random (randomRs, newStdGen)

toVector :: [String] -> Vector 
toVector rows = map (\c -> if c == '#' then 1.0 else 0.0) (concat rows)

digitTemplate :: Int -> [String]
digitTemplate 0 = 
  [ " ### "
  , "#   #"
  , "#   #"
  , "#   #"
  , " ### " ]
digitTemplate 1 = 
  [ "  #  "
  , " ##  "
  , "  #  "
  , "  #  "
  , " ### " ]
digitTemplate 2 = 
  [ " ### "
  , "    #"
  , " ### "
  , "#    "
  , " ### " ]
digitTemplate 3 = 
  [ " ### "
  , "    #"
  , " ### "
  , "    #"
  , " ### " ]
digitTemplate 4 = 
  [ "#   #"
  , "#   #"
  , " #####"
  , "    #"
  , "    #" ]
digitTemplate 5 = 
  [ "#####"
  , "#    "
  , "#### "
  , "    #"
  , "#### " ]
digitTemplate 6 = 
  [ " ### "
  , "#    "
  , "#### "
  , "#   #"
  , " ### " ]
digitTemplate 7 = 
  [ "#####"
  , "   # "
  , "  #  "
  , " #   "
  , "#    " ]
digitTemplate 8 = 
  [ " ### "
  , "#   #"
  , " ### "
  , "#   #"
  , " ### " ]
digitTemplate 9 = 
  [ " ### "
  , "#   #"
  , " ####"
  , "    #"
  , " ### " ]
digitTemplate _ = replicate 5 "     "


padTo28 :: [String] -> [String]
padTo28 smallGrid = 
  let topPad    = replicate 11 "                            "
      bottomPad = replicate 12 "                            "
      padRow r  = "###########" ++ r ++ "############"
  in topPad ++ map padRow smallGrid ++ bottomPad

getPerfectDigit :: Int -> Vector 
getPerfectDigit n = toVector $ padTo28 (digitTemplate n)

addNoise :: Double -> Vector -> IO Vector
addNoise level clear = do
 gen <- newStdGen
 let randomDeltas = randomRs (-level, level) gen
 return $ zipWith (\p noise -> max 0.0 (min 1.0 (p + noise))) clear randomDeltas

generateSample :: Int -> Double -> IO Image 
generateSample digit noise = do
  let cleanImg = getPerfectDigit digit
  noisyImg <- addNoise noise cleanImg 
  return (noisyImg, digit)

generateDataset :: Int -> Double -> IO [Image]
generateDataset size noise = mapM (\idx -> generateSample (idx `mod` 10) noise) [1..size]
