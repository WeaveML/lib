module CLI where

import Options.Applicative

data Command
  = Train { epochs :: Int, output :: FilePath }
  | Update { epochs :: Int, input :: FilePath }
  | Predict { inputModel :: FilePath, imageFile :: FilePath }
  deriving (Show)

trainParser :: Parser Command
trainParser = Train
  <$> option auto
      ( long "train"
     <> short 't'
     <> metavar "EPOCHS"
     <> help "Train model with entered epochs" )
  <*> strOption
      ( long "output"
     <> short 'o'
     <> metavar "FILE"
     <> help "Path for saved model file" )

updateParser:: Parser Command
updateParser = Update
  <$> option auto
      ( long "update"
     <> short 'u'
     <> metavar "EPOCHS"
     <> help "Retrain model with entered epochs" )
  <*> strOption
      ( long "output"
     <> short 'o'
     <> metavar "FILE"
     <> help "Path for saved model file" )


predictParser :: Parser Command
predictParser = Predict
  <$> strOption
      ( long "input"
     <> short 'I'
     <> metavar "MODEL_FILE"
     <> help "Path for loaded model file" )
  <*> strOption
      ( long "image"
     <> short 'i'
     <> metavar "IMAGE_FILE"
     <> help "Path for binary file (784b)" )

actionParser :: Parser Command
actionParser = trainParser <|> predictParser <|> updateParser

optsInfo :: ParserInfo Command
optsInfo = info (actionParser <**> helper)
  ( fullDesc
 <> progDesc "Weave-Image: cli interface for weave-image ml models"
 <> header "weave - haskell ml" )

parseCLI :: IO Command
parseCLI = execParser optsInfo
