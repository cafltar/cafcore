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
        qcAppliedColName = col + qcAppliedSuffix
        qcResultColName = col + qcResultSuffix
        qcPhraseColName = col + qcPhraseColSuffix
        
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

def quality_assurance_calculated(df: pd.DataFrame, idColName: str, parentCols: list) -> pd.DataFrame:
    """Creates a new DataFrame from df and sets the QA bit based on the QA bit from a list of parent columns
    """
    
    qcAppliedSuffix = '_qcApplied'
    qcResultSuffix = '_qcResult'
    qcPhraseSuffix = '_qcPhrase'

    result = df.copy()

    # Make sure parentCols exist
    for col in parentCols:
        qcResultCol = col + qcResultSuffix
        if qcResultCol not in df.columns:
            raise Exception("Could not find parent column: " + qcResultCol)

    # Check parentCols if any QC failed
    for index, row in result.iterrows():
        numberColsFail = 0
        reasonPhrase = ""

        colApplied = idColName + qcAppliedSuffix
        colResult = idColName + qcResultSuffix
        colPhrase = idColName + qcPhraseSuffix

        for col in parentCols:
            qcResultCol = col + qcResultSuffix
            #if int(row[qcResultCol] != None):
            if not pd.isna(row[qcResultCol]):
                if int(row[qcResultCol]) > 0:
                    numberColsFail += 1
                    qcPhraseCol = col + qcPhraseSuffix

                    if reasonPhrase == "":
                        reasonPhrase = (col + ": " + row[qcPhraseCol])
                    else:
                        reasonPhrase = reasonPhrase + " , " + (col + ": " + row[qcPhraseCol])

        if numberColsFail > 0:
            reasonPhrase = "(Assurance) values used in a calculation failed one or more QC checks: [" + reasonPhrase + "]"

            result.at[index, colApplied] = update_qc_bitstring(row[colApplied], "000001")
            result.at[index, colResult] = update_qc_bitstring(row[colResult], "000001")
            result.at[index, colPhrase] = update_phrase(row[colPhrase], reasonPhrase)
        else:
            result.at[index, colApplied] = update_qc_bitstring(row[colApplied], "000001")
            result.at[index, colResult] = update_qc_bitstring(row[colResult], "000000")

    return result

def quality_assurance(df: pd.DataFrame, pathToQAFile: str, idColName: str) -> pd.DataFrame:
    """Creates a new DataFrame from df with changes (Create, Update, Delete, Flag) specified in the specified QA file

    :param df: DataFrame, must have column that matches "idColName"
    :param pathToQAFile: str, path to existing csv file with columns: "ID", "Verb", "Variable", "NewVal", "Comment", "Reviewer"
    :param idColName: str, the name of the column in df that has values specified in the "ID" column in the QA file
    :rtype: pd.DataFrame, copy of df with changes specified in QA file
    """

    qa = pd.read_csv(pathToQAFile)

    qaDelete = qa[qa["Verb"] == "Delete"]
    qaCreate = qa[qa["Verb"] == "Create"]
    qaUpdate = qa[qa["Verb"] == "Update"]
    qaFlag = qa[qa["Verb"] == "Flag"]

    result = df.copy()

    # Delete values
    for index, row in qaDelete.iterrows():
        matchedRowIndex = result[result[idColName] == row["ID"]].index

        if len(matchedRowIndex) > 1:
            raise Exception("Multiple values found for given ID, check input dataframe")

        result.drop(matchedRowIndex, inplace = True)

    result = result.reset_index(drop=True)

    # Create a new row
    for index, row in qaCreate.iterrows():
        rowValues = eval(row["NewVal"])
        newRow = pd.DataFrame(rowValues, index=[0])

        result = pd.concat([result, newRow], axis = 0, ignore_index=True)

        dfCols = list(result.columns)
        newRowCols = list(newRow.columns)

        # Update _qcPhrase columns if they exist in the main dataset
        for c in newRowCols:
            potentialQCCol = c + "_qcPhrase"
            if potentialQCCol in dfCols:
                changePhrase = "(Assurance) Row created, reason: {}".format(row["Comment"])

                series = result.loc[(result[idColName] == row["ID"]), potentialQCCol]

                result.loc[(result[idColName] == row["ID"]), potentialQCCol] = update_phrase(
                    result.loc[series.index[0], potentialQCCol],
                    changePhrase
                )

    result = result.reset_index(drop=True)

    # Update values
    for index, row in qaUpdate.iterrows():
        
        prevValueSeries = result.loc[(result[idColName].astype(str) == row["ID"]), row["Variable"]]
        
        if len(prevValueSeries) == 0:
            raise Exception("ID not found, check QA File")
        if len(prevValueSeries) > 1:
            raise Exception("Multiple values found for given ID, check input dataframe")
        
        prevValue = prevValueSeries.values[0]
        changePhraseCol = row["Variable"] + "_qcPhrase"

        if(pd.isna(row["NewVal"])):
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

    # Flag values by setting QA bit to fail
    for index, row in qaFlag.iterrows():
        reasonPhraseCol = row['Variable'] + '_qcPhrase'
        changePhrase = "(Assurance) {}".format(row['Comment'])

        qaResultCol = row['Variable'] + '_qcResult'

        # Create column if not exist
        if reasonPhraseCol not in result.columns:
            result[reasonPhraseCol] = None
        if qaResultCol not in result.columns:
            result[qaResultCol] = None

        phrase_series = result.loc[(result[idColName] == row['ID']), reasonPhraseCol]

        result.loc[(result[idColName] == row['ID']), reasonPhraseCol] = update_phrase(
            result.loc[phrase_series.index[0], reasonPhraseCol],
            changePhrase
        )

        result.loc[(result[idColName] == row['ID']), qaResultCol] = update_qc_bitstring(
            '000001', '000000')

    return result

def process_qc_bounds_check(
        df: pd.DataFrame, 
        idColName: str, 
        lower: int, 
        upper: int,
        flagNulls: bool = False) -> pd.DataFrame:
    """Conducts a bounds check for each measurement in the idColName specified, adds _qcApplied, _qcResult, and _qcPhrase columns. Sets the "point" bit to true for _qcApplied ("000010"), updates _qcResult depending on pass/fail, and specifies reason for fail in _qcPhrase.

    If QC columns not present, ones will be created.

    :param df: pd.DataFrame, the dataframe with values in idColName to be compared to lower and upper
    :param idColName: str, name of the column in df to be checked
    :param lower: int, the lower bounds of the bounds check
    :param upper: int, the upper bounds of the bounds check
    :param flagNulls: bool, if true then null values will fail the bounds check
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
    # TODO: Definitely refactor this ugly code
    if(flagNulls):
        df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] < lower) | (df_out[idColName] > upper)), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000010")))
    
        changePhrase = "(Point) Value outside of bounds [{0}, {1}]".format(lower, upper)
        df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] < lower) | (df_out[idColName] > upper)), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))
    else:
        df_out.update(df_out.loc[((df_out[idColName] < lower) | (df_out[idColName] > upper)), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000010")))
    
        changePhrase = "(Point) Value outside of bounds [{0}, {1}]".format(lower, upper)
        df_out.update(df_out.loc[((df_out[idColName] < lower) | (df_out[idColName] > upper)), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))

    return df_out

def process_qc_greater_than_check(
        df: pd.DataFrame,
        idColName: str,
        idColNameCompare: str,
        flagNulls: bool = False) -> pd.DataFrame:
    """Conducts a quality control check at the observation level. Checks if idColName is greater than idColNameCompare. Test fails if not.
    :rtype: pd.DataFrame, dataframe copied from df with additional columns and results of the bounds check
    """

    df_out = df.copy()

    qc_applied_colName = idColName + "_qcApplied"
    qc_result_colName = idColName + "_qcResult"
    qc_reason_colName = idColName + "_qcPhrase"
    
    if qc_applied_colName not in df.columns:
        df_out[qc_applied_colName] = "000000"
    
    df_out[qc_applied_colName] = df_out[qc_applied_colName].apply(lambda x: update_qc_bitstring(x, "000100"))

    if qc_result_colName not in df.columns:
        df_out[qc_result_colName] = "000000"

    # result code defaults to 0 (passed), set to 000100 (fail) if not greater than
    # TODO: Definitely refactor this ugly code
    if(flagNulls):
        df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] < df_out[idColNameCompare])), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000100")))
    
        changePhrase = "(Observation) Value should be greater than value for {0}".format(idColNameCompare)
        df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] < df_out[idColNameCompare])), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))
    else:
        df_out.update(df_out.loc[((df_out[idColName] < df_out[idColNameCompare])), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000100")))
    
        changePhrase = "(Observation) Value should be greater than value for {0}".format(idColNameCompare)
        df_out.update(df_out.loc[((df_out[idColName] < df_out[idColNameCompare])), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))

    return df_out

def process_qc_less_than_check(
        df: pd.DataFrame,
        idColName: str,
        idColNameCompare: str,
        flagNulls: bool = False) -> pd.DataFrame:
    """Conducts a quality control check at the observation level. Checks if idColName is less than idColNameCompare. Test fails if not.
    :rtype: pd.DataFrame, dataframe copied from df with additional columns and results of the bounds check
    """

    df_out = df.copy()

    qc_applied_colName = idColName + "_qcApplied"
    qc_result_colName = idColName + "_qcResult"
    qc_reason_colName = idColName + "_qcPhrase"
    
    if qc_applied_colName not in df.columns:
        df_out[qc_applied_colName] = "000000"
    
    df_out[qc_applied_colName] = df_out[qc_applied_colName].apply(lambda x: update_qc_bitstring(x, "000100"))

    if qc_result_colName not in df.columns:
        df_out[qc_result_colName] = "000000"

    # result code defaults to 0 (passed), set to 000100 (fail) if not greater than
    # TODO: Definitely refactor this ugly code
    if(flagNulls):
        df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] > df_out[idColNameCompare])), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000100")))
    
        changePhrase = "(Observation) Value should be less than value for {0}".format(idColNameCompare)
        df_out.update(df_out.loc[(pd.isna(df_out[idColName]) | (df_out[idColName] > df_out[idColNameCompare])), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))
    else:
        df_out.update(df_out.loc[((df_out[idColName] > df_out[idColNameCompare])), qc_result_colName].apply(lambda x: update_qc_bitstring(x, "000100")))
    
        changePhrase = "(Observation) Value should be less than value for {0}".format(idColNameCompare)
        df_out.update(df_out.loc[((df_out[idColName] > df_out[idColNameCompare])), qc_reason_colName].apply(lambda x: update_phrase(x, changePhrase)))

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

def ensure_columns(df: pd.DataFrame, dimension_cols: list, metric_cols: list, check_qc_cols:bool = True) -> bool:
    """Checks that df only has the columns in the dimension_cols and metric_cols; includes associated qc columns by default

    param df: pd.DataFrame, dataframe with the columns to be checked
    param dimension_cols
    """
    qc_suffixes = ['_qcApplied', '_qcResult', '_qcPhrase']
    all_cols = df.columns

    non_qc_cols = [col for col in all_cols if not any(qc_suffix in col for qc_suffix in qc_suffixes)]
    qc_cols = [col for col in all_cols if any(qc_suffix in col for qc_suffix in qc_suffixes)]

    check_passes = True

    # Ensure that column counts are as expected
    if check_qc_cols:
        expected_num_cols = len(dimension_cols) + len(metric_cols) + (len(qc_suffixes) * len(metric_cols))
        if expected_num_cols != len(all_cols):
            return False
    else:
        expected_num_cols = len(dimension_cols) + len(metric_cols)
        if expected_num_cols != len(non_qc_cols):
            return False
        
    # Ensure that all expected columns are present
    if set(non_qc_cols) != set(dimension_cols + metric_cols):
        return False
        
    if check_qc_cols:
        expected_qc_cols = [metric_col + qc_suffix for metric_col in metric_cols for qc_suffix in qc_suffixes]
        if set(qc_cols) != set(expected_qc_cols):
            return False
        
    return check_passes

def add_processing_suffix(df: pd.DataFrame, col_basenames: list, processing_level: int, include_qc_cols: bool = True) -> pd.DataFrame:
    """Adds P{processing_level} to the end of all metric columns and updates relevant QC columns if indicated
    """
    result = df.copy()
    processing_suffix = '_P' + str(processing_level)
    qc_suffixes = ['_qcApplied', '_qcResult', '_qcPhrase']

    for col in col_basenames:
        if col in result.columns:
            result.rename(columns = {col: col+processing_suffix}, inplace = True)

            if include_qc_cols:
                for qc_suffix in qc_suffixes:
                    qc_col = col + qc_suffix
                    if qc_col in result.columns:
                        new_qc_col = col + processing_suffix + qc_suffix
                        result.rename(columns = {qc_col: new_qc_col}, inplace = True)

    return result

#def ensure_rows_by_timestep(df: pd.DataFrame, datetime_col:str, observation_col: str, observation_ids: list, timestep: pd.Timedelta):
#    """Checks that each observation is present for each timestep
#    """
#    raise Exception("Not implemented")
