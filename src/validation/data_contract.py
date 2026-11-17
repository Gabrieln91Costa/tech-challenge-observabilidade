
import pandas as pd
import pandera.pandas as pa

from pathlib import Path
from pandera.errors import SchemaError


# =====================================================
# CONFIGURAÇÕES
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR / "data" / "reference" / "credit_reference.csv"
)

INVALID_DIR = BASE_DIR / "data" / "invalid"

INVALID_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# CONTRATO DE QUALIDADE DOS DADOS
# =====================================================

credit_schema = pa.DataFrameSchema(
    {
        "SeriousDlqin2yrs": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False
        ),

        "RevolvingUtilizationOfUnsecuredLines": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "age": pa.Column(
            int,
            checks=[
                pa.Check.ge(18),
                pa.Check.le(100)
            ],
            nullable=False
        ),

        "NumberOfTime30-59DaysPastDueNotWorse": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "DebtRatio": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "MonthlyIncome": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "NumberOfOpenCreditLinesAndLoans": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "NumberOfTimes90DaysLate": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "NumberRealEstateLoansOrLines": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "NumberOfTime60-89DaysPastDueNotWorse": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False
        ),

        "NumberOfDependents": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False
        ),
    },

    strict=True,
    coerce=True
)


# =====================================================
# VALIDAÇÃO DOS DADOS
# =====================================================

def validate_credit_data(
    df: pd.DataFrame,
    check_duplicates: bool = True
):

    # Verificar duplicidades originais
    if check_duplicates and df.duplicated().any():

        duplicated_count = df.duplicated().sum()

        raise ValueError(
            f"Existem {duplicated_count} "
            "linhas duplicadas no dataset."
        )

    # Validar regras do contrato
    validated_df = credit_schema.validate(df)

    return validated_df


# =====================================================
# PREPARAÇÃO DOS DADOS
# =====================================================

def prepare_reference_data(df: pd.DataFrame):

    df = df.copy()

    # 1. Identificar idades inválidas
    invalid_age_mask = (
        (df["age"] < 18) |
        (df["age"] > 100)
    )

    df_invalid_age = df[invalid_age_mask].copy()

    df_valid = df[~invalid_age_mask].copy()

    # 2. Identificar duplicidades originais
    duplicate_mask = df_valid.duplicated(keep="first")

    df_duplicates = df_valid[duplicate_mask].copy()

    # Manter uma ocorrência de cada registro
    df_unique = df_valid.drop_duplicates(
        keep="first"
    ).copy()

    # 3. Criar indicadores de valores ausentes
    for column in ["MonthlyIncome", "NumberOfDependents"]:

        df_unique[f"{column}_was_missing"] = (
            df_unique[column].isnull().astype(int)
        )

    # 4. Imputar valores ausentes pela mediana
    columns_to_impute = [
        "MonthlyIncome",
        "NumberOfDependents"
    ]

    for column in columns_to_impute:

        median_value = df_unique[column].median()

        df_unique[column] = df_unique[column].fillna(
            median_value
        )

    # Remover indicadores temporariamente para validar
    # as 11 colunas originais do contrato.
    df_clean = df_unique.drop(
        columns=[
            "MonthlyIncome_was_missing",
            "NumberOfDependents_was_missing"
        ]
    )

    return df_clean, df_invalid_age, df_duplicates


# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================

if __name__ == "__main__":

    print("\n========== CARREGANDO DATASET ==========")

    df = pd.read_csv(DATASET_PATH)

    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    print(f"Registros originais: {len(df)}")

    # -------------------------------------------------
    # TESTE 1 - DADOS ORIGINAIS
    # -------------------------------------------------

    print("\n========== TESTE 1: DADOS ORIGINAIS ==========")

    try:

        validate_credit_data(df)

        print("CONTRATO APROVADO!")

    except (SchemaError, ValueError) as error:

        print("CONTRATO REPROVADO!")
        print(f"Motivo: {error}")

    # -------------------------------------------------
    # TESTE 2 - PREPARAÇÃO DOS DADOS
    # -------------------------------------------------

    print("\n========== TESTE 2: DADOS TRATADOS ==========")

    df_clean, df_invalid_age, df_duplicates = (
        prepare_reference_data(df)
    )

    # Salvar registros com idade inválida
    df_invalid_age.to_csv(
        INVALID_DIR / "credit_invalid_age.csv",
        index=False
    )

    # Salvar duplicidades excedentes originais
    df_duplicates.to_csv(
        INVALID_DIR / "credit_duplicates.csv",
        index=False
    )

    print(
        f"Registros rejeitados por idade: "
        f"{len(df_invalid_age)}"
    )

    print(
        f"Duplicidades excedentes: "
        f"{len(df_duplicates)}"
    )

    print(
        f"Registros preparados: {len(df_clean)}"
    )

    try:

        # Duplicidades já foram tratadas antes da imputação.
        # Não verificar novamente as duplicidades artificiais.
        validate_credit_data(
            df_clean,
            check_duplicates=False
        )

        print("\nCONTRATO APROVADO!")
        print(f"Registros validados: {len(df_clean)}")
        print("Valores nulos tratados.")
        print("Duplicidades originais tratadas.")
        print("Regras de qualidade aprovadas.")

        # Salvar dataset aprovado
        df_clean.to_csv(
            BASE_DIR / "data" / "reference" /
            "credit_reference_clean.csv",
            index=False
        )

    except (SchemaError, ValueError) as error:

        print("\nFALHA NA VALIDAÇÃO!")
        print(f"Detalhes: {error}")

    # -------------------------------------------------
    # TESTE 3 - SIMULAÇÃO DE DADOS INVÁLIDOS
    # -------------------------------------------------

    print("\n========== TESTE 3: DADOS INVÁLIDOS ==========")

    df_test_invalid = df_clean.copy()

    # Forçar idade inválida
    df_test_invalid.loc[
        df_test_invalid.index[0],
        "age"
    ] = 15

    # Forçar renda nula
    df_test_invalid.loc[
        df_test_invalid.index[1],
        "MonthlyIncome"
    ] = None

    try:

        validate_credit_data(
            df_test_invalid,
            check_duplicates=False
        )

        print("ERRO: dados inválidos foram aprovados!")

    except (SchemaError, ValueError):

        print("CONTRATO REPROVADO!")
        print("Idade inválida e renda nula identificadas.")
        print("Ingestão bloqueada com sucesso.")

    print("\n========== PROCESSAMENTO FINALIZADO ==========")
