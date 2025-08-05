# utils/preprocessing.py



def preprocess_data(df):
    """
    Preprocess Spark DataFrame by dropping rows with missing ReportText.
    Returns cleaned list of report strings.
    """
    cleaned_df = df.dropna(subset=["ReportText"])
    return cleaned_df
