module Weave.IO (saveModel, loadModel) where

import Weave.Base (Network(..))
import Data.Binary (encodeFile, decodeFile)

saveModel :: Network -> FilePath -> IO ()
saveModel net path = encode path net 

loadModel :: FilePath -> IO Network
loadModel path = decodeFile path
