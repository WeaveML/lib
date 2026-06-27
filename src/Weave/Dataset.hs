module Weave.Dataset
  ( generateDataset
  , generateSample
  ) where

import Weave.Base (Vector, Image, imgW, imgH)
import System.Random (randomRs, newStdGen)

digitTemplate :: Int -> [String]
digitTemplate 0 =
  [ "  ######  "
  , " ##    ## "
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , " ##    ## "
  , "  ######  " ]
digitTemplate 1 =
  [ "    ##    "
  , "   ###    "
  , "  ####    "
  , "    ##    "
  , "    ##    "
  , "    ##    "
  , "    ##    "
  , "    ##    "
  , "    ##    "
  , "  ######  " ]
digitTemplate 2 =
  [ "  ######  "
  , " ##    ## "
  , "        ##"
  , "        ##"
  , "    ####  "
  , "   ##     "
  , "  ##      "
  , " ##       "
  , "##        "
  , "##########" ]
digitTemplate 3 =
  [ "  ######  "
  , " ##    ## "
  , "        ##"
  , "        ##"
  , "   #####  "
  , "        ##"
  , "        ##"
  , "        ##"
  , " ##    ## "
  , "  ######  " ]
digitTemplate 4 =
  [ "##        "
  , "##    ##  "
  , "##    ##  "
  , "##    ##  "
  , "##########"
  , "      ##  "
  , "      ##  "
  , "      ##  "
  , "      ##  "
  , "      ##  " ]
digitTemplate 5 =
  [ "##########"
  , "##        "
  , "##        "
  , "##        "
  , " #######  "
  , "        ##"
  , "        ##"
  , "        ##"
  , " ##    ## "
  , "  ######  " ]
digitTemplate 6 =
  [ "  ######  "
  , " ##    ## "
  , "##        "
  , "##        "
  , "########  "
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , " ##    ## "
  , "  ######  " ]
digitTemplate 7 =
  [ "##########"
  , "        ##"
  , "       ## "
  , "      ##  "
  , "     ##   "
  , "    ##    "
  , "   ##     "
  , "   ##     "
  , "   ##     "
  , "   ##     " ]
digitTemplate 8 =
  [ "  ######  "
  , " ##    ## "
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , " ####### "
  , "##      ##"
  , "##      ##"
  , "##      ##"
  , " ##    ## "
  , "  ######  " ]
digitTemplate 9 =
  [ "  ######  "
  , " ##    ## "
  , "##      ##"
  , "##      ##"
  , " #########"
  , "        ##"
  , "        ##"
  , "        ##"
  , " ##    ## "
  , "  ######  " ]
digitTemplate _ = replicate 10 (replicate 10 ' ')

templateH, templateW :: Int
templateH = 10
templateW = 10

padToCanvas :: [String] -> [String]
padToCanvas small =
  let topPad    = replicate topRows emptyRow
      bottomPad = replicate botRows emptyRow
      leftPad   = replicate leftCols ' '
      rightPad  = replicate rightCols ' '
      topRows   = (imgH - templateH) `div` 2
      botRows   = imgH - templateH - topRows
      leftCols  = (imgW - templateW) `div` 2
      rightCols = imgW - templateW - leftCols
      emptyRow  = replicate imgW ' '
      padRow r  = leftPad ++ r ++ rightPad
  in topPad ++ map padRow small ++ bottomPad

toVector :: [String] -> Vector
toVector rows = map (\c -> if c == '#' then 1.0 else 0.0) (concat rows)

getPerfectDigit :: Int -> Vector
getPerfectDigit n = toVector $ padToCanvas (digitTemplate n)

addNoise :: Double -> Vector -> IO Vector
addNoise level clear = do
  gen <- newStdGen
  let deltas = randomRs (-level, level) gen
  return $ zipWith (\p d -> max 0.0 (min 1.0 (p + d))) clear deltas

-- pbulci api
generateSample :: Int -> Double -> IO Image
generateSample digit noise = do
  let cleanImg = getPerfectDigit digit
  noisyImg <- addNoise noise cleanImg
  return (noisyImg, digit)

generateDataset :: Int -> Double -> IO [Image]
generateDataset size noise =
  mapM (\i -> generateSample (i `mod` 10) noise) [1 .. size]
