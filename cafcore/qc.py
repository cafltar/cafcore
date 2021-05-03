import pandas as pd
import numpy as np

def initialize_qc(df: pd.DataFrame, colOmit:list = []) -> pd.DataFrame:
    """Creates three quality control columns by applying suffixes ("_qcApplied", "_qcResult", "_qcPhrase") to  each column in the passed DataFrame not in colOmit. Ignores existing QC columns.
    
    :param df: DataFrame that is copied and new columns are added to
    :param colOmit: List of strings, that correspond to df column names, where no QC columns should be created
    :rtype: DataFrame that is a copy of df with added columns.
    """

    qcAppliedSuffix = "_qcApplied"
    qcResultSuffix = "_qcResult"
    qcPhraseColSuffix = "_qcPhrase"

    # Filter out qcColumns and any columns specified in colOmit
    colNames = [c for c in df.columns if c not in colOmit if qcAppliedSuffix not in c if qcResultSuffix not in c if qcPhraseColSuffix not in c]


    result = df.copy()

    for col in colNames:
        qcAppliedColName = col + "_qcApplied"
        qcResultColName = col + "_qcResult"
        qcPhraseColName = col + "_qcPhrase"
        
        if qcAppliedColName not in result.columns:
            result[qcAppliedColName] = "000000"
        
        if qcResultColName not in result.columns:
            result[qcResultColName] = "000000"
        
        if qcPhraseColName not in result.columns:
            result[qcPhraseColName] = ""
    
    return result

def update_qc_bitstring(currentQCCode:str, qcCode:str) -> str:
    """Updates a six bit bitstring in the form of "000000". For example, "000001" added to "000010" will result in "000011". A null currentQCCode is set to qcCode.

    :param currentQCCode: str, the current bitstring to be updated
    :param qcCode: str, the bitstring to be added to the currentQCCode
    :rtype: str, the new bitstring
    """

    # Assume pd.isnull is undefined so initialize it with the qcCode value
    if pd.isnull(currentQCCode) or (not currentQCCode):
        return qcCode

    if(len(currentQCCode) != 6 | len(qcCode) != 6):
        raise Exception("Binary strings not equal length and do not equal 6 bits")

    newQCCode = ""

    for i in range(len(currentQCCode)):
        newQCCode += str(int(bool(int(currentQCCode[i])) | bool(int(qcCode[i]))))
    
    return newQCCode

def update_phrase(currentPhrase: str, phrase:str) -> str:
    """Updates a qcPhrase by appending phrase to currentPhrase separated by " | ". A null currentPhrase will be set to phrase.

    :param currentPhrase: str, the current phrase string
    :param phrase: str, the phrase to be concatenated
    :rtype: str, the combined phrase
    """
    
    # Assume pd.isnull is undefined so initialize it with the passed phrase
    if pd.isnull(currentPhrase) or (not currentPhrase):
        newPhrase = phrase
    else:
        newPhrase = currentPhrase + " | " + phrase

    return newPhrase

def set_quality_assurance_applied(df: pd.DataFrame, colsOmit:list = [], qcApplied:bool = True, qcPassed:bool = True) -> pd.DataFrame:
    """Sets the quality assurance bit for all "_qcApplied" and "_qcResult" columns in df

    :param df: DataFrame with "_qcApplied" and "_qcResult" columns, returns a copy
    :param colsOmit: a list of column names in df to ignore
    :param qcApplied: bool, sets applied bit to 1 if true, 0 if false
    :param qcPassed: bool, sets result bit to 0 if true, 1 if false
    :rtype: DataFrame. 
    """

    qcAppliedSuffix = "_qcApplied"
    qcResultSuffix = "_qcResult"

    colNamesApplied = [c for c in df.columns if c not in colsOmit if qcAppliedSuffix in c]
    colNamesResult = [c for c in df.columns if c not in colsOmit if qcResultSuffix in c]

    result = df.copy()

    appliedBitString = "000001" if qcApplied else "000000"
    resultBitString = "000000" if qcPassed else "000001"

    for col in colNamesApplied:
        result[col] = result[col].apply(lambda x: update_qc_bitstring(x, appliedBitString))

    for col in colNamesResult:
        result[col] = result[col].apply(lambda x: update_qc_bitstring(x, resultBitString))

    return result

def quality_assurance(df: pd.DataFrame, pathToQAFile: str, idColName: str) -> pd.DataFrame:
    """Creates a new DataFrame from df with changes (Create, Update, Delete) specified in the specified QA file

    :param df: DataFrame, must have column that matches "idColName"
    :param pathToQAFile: str, path to existing csv file with columns: "ID", "Verb", "Variable", "NewVal", "Comment", "Reviewer"
    :param idColName: str, the name of the column in df that has values specified in the "ID" column in the QA file
    :rtype: pd.DataFrame, copy of df with changes specified in QA file
    """

    qa = pd.read_csv(pathToQAFile)

    qaDelete = qa[qa["Verb"] == "Delete"]
    qaCreate = qa[qa["Verb"] == "Create"]
    qaUpdate = qa[qa["Verb"] == "Update"]

    result = df.copy()

    # Delete values
    for index, row in qaDelete.iterrows():
        matchedRowIndex = result[result[idColName] == row["ID"]].index

        if len(matchedRowIndex) > 1:
            raise Exception("Multiple values found for given ID, check input dataframe")

        result.drop(matchedRowIndex, inplace = True)

    for index, row in qaCreate.iterrows():
        raise Exception("Not implemented yet")

    # Update values
    for index, row in qaUpdate.iterrows():
        
        prevValueSeries = result.loc[(result[idColName] == row["ID"]), row["Variable"]]
        
        if len(prevValueSeries) == 0:
            raise Exception("ID not found, check QA File")
        if len(prevValueSeries) > 1:
            raise Exception("Multiple values found for given ID, check input dataframe")
        
        prevValue = prevValueSeries.values[0]
        changePhraseCol = row["Variable"] + "_qcPhrase"

        if(np.isnan(row["NewVal"])):
            result.loc[(result[idColName] == row["ID"]), row["Variable"]] = None
        else:
            result.loc[(result[idColName] == row["ID"]), row["Variable"]] = row["NewVal"]

        changePhrase = "(Assurance) Previous val: {}, reason: {}".format(prevValue, row["Comment"])
        
        # Create column if not exist
        if changePhraseCol not in result.columns:
            result[changePhraseCol] = None

        series = result.loc[(result[idColName] == row["ID"]), changePhraseCol]

        result.loc[(result[idColName] == row["ID"]), changePhraseCol] = update_phrase(
            result.loc[series.index[0], changePhraseCol],
            changePhrase
        )

    return result

def process_qc_bounds_check(df: pd.DataFrame, idColName: str, lower: int, upper: int) -> pd.DataFrame:
    """Conducts a bounds check for each measurement in the idColName specified, adds _qcApplied, _qcResult, and _qcPhrase columns. Sets the "point" bit to true for _qcApplied ("000010"), updates _qcResult depending on pass/fail, and specifies reason for fail in _qcPhrase.

    If QC columns not present, ones will be created.

    :param df: pd.DataFrame, the dataframe with values in idColName to be compared to lower and upper
    :param idColName: str, name of the column in df to be checked
    :param lower: int, the lower bounds of the bounds check
    :param upper: int, the upper bounds of the bounds check
    :rtype: pd.DataFrame, dataframe copied from df with additional columns and results of the bounds check
    """

    df_out = df.copy()

    qc_applied_colName = idColName + "_qcApplied"
    qc_result_colName = idColName + "_qcResult"
    qc_reason_colName = idColName + "_qcPhrase"

    # TODO: Need to handle adding previous QC checks - use bin() / bit manip
    # 000010 means level 2 check, or check operating on single value
    
    if qc_applied_colName not in df.columns:
        df_out[qc_applied_colName] = "000000"
    
    df_out[qc_applied_colName] = df_out[qc_applied_colName].apply(lambda x: update_qc_bitstring(x, "000010"))

    if qc_result_colName not in df.columns:
        df_out[qc_result_colName] = "000000"

    # result code defaults to 0 (passed), set to 000010 (fail) if outside of bounds
    df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] < lower) | (df_out[idColName] > upper)), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000010")))
    
    changePhrase = "(Point) Value outside of bounds [{0}, {1}]".format(lower, upper)
    df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] < lower) | (df_out[idColName] > upper)), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))
    
    return df_out

def sort_qc_columns(df: pd.DataFrame, groupWithMeasure:bool = True) -> pd.DataFrame:
    """Sorts the columns in df by either grouping the QC columns with the measurement column (groupWithMeasure = true) or listing the QC columns after the measurement columns (groupWithMeasure = False)

    :param df: pd.DataFrame, dataframe with columns to be rearranged
    :param groupWithMeasure: bool, default = True, places QC columns along with corresponding measurement column if True, places at end if False
    :rtype: pd.DataFrame, the sorted DataFrame
    """

    dfOut = df.copy()

    cols = dfOut.columns.tolist()
    colsQC = [x for x in cols if "_" in x]
    colMeasure = [x for x in cols if "_" not in x]

    sortedCols = []

    if groupWithMeasure:

        for baseName in colMeasure:
            sortedCols.append(baseName)
            
            colsQCMatchBase = [x for x in colsQC if baseName == x.split("_")[0]]
            sortedCols = sortedCols + colsQCMatchBase
    else:

        sortedCols = colMeasure + colsQC

    return dfOut[sortedCols]
