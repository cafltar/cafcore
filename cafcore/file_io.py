import pandas as pd
import pathlib
import datetime

def write_data_csv(
    df:pd.DataFrame, 
    pathToOutputDir:pathlib.Path, 
    fileBasename:str, 
    processingLevel:int = None, 
    accuracyLevel:int = None) -> None:

    """Writes a csv file to given directory with the following structure: {baseName}_{P1}{A1}_{YYYYMMDD}.csv. A directory will be created if needed.

    :param df: pd.DataFrame to be written as a CSV file
    :param pathToOutputDir: pathlib.Path where CSV file is written. Directory will be created if it does not exist
    :param fileBasename: str, base name of the filename
    :param processingLevel: int, default = None, Processing level of the dataset
    :param accuracyLevel: int, default = None, Accuracy level of the dataset
    """
    
    # Create iso 8601 date string for file versioning
    date_today = datetime.datetime.now().strftime("%Y%m%d")

    fileName = fileBasename

    # Set processing and accuracy level, if needed
    if (accuracyLevel != None or processingLevel != None):
        fileName = fileName + "_"

        if processingLevel != None:
            fileName = fileName + "P" + str(processingLevel)

        if accuracyLevel != None:
            fileName = fileName + "A" + str(accuracyLevel)
        
    # Tie it all together
    fileName = fileName + "_" + date_today + ".csv"

    # Create the dir path, if needed
    pathToOutputDir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        (pathToOutputDir / fileName),
        index = False
    )    
