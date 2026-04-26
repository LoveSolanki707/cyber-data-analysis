import pandas as pd

def clean_data(path):
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna()

    print("Columns:", df.columns)

    return df