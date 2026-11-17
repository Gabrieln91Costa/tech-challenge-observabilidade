
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "reference"
    / "credit_reference.csv"
)


def load_reference_data():

    df = pd.read_csv(DATASET_PATH)

    # Remover coluna de índice original do Kaggle
    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    print("\n========== DATASET DE REFERÊNCIA ==========")

    print(f"\nRegistros: {len(df)}")
    print(f"Colunas: {len(df.columns)}")

    print("\n========== COLUNAS ==========")
    print(df.columns.tolist())

    print("\n========== VALORES NULOS ==========")
    print(df.isnull().sum())

    print("\n========== DISTRIBUIÇÃO DO TARGET ==========")
    print(df["SeriousDlqin2yrs"].value_counts())

    print("\n========== TIPOS DE DADOS ==========")
    print(df.dtypes)

    return df


if __name__ == "__main__":
    load_reference_data()
