from scripts.clean import clean_data
from scripts.train import train_model

def run_pipeline():
    df = clean_data("data/cyber.csv")

    score = train_model(df)

    print("Model Accuracy:", score)

if __name__ == "__main__":
    run_pipeline()