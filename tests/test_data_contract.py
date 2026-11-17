import pandas as pd
import pytest

from pandera.errors import SchemaError

from src.validation.data_contract import (
    validate_credit_data,
    prepare_reference_data
)


@pytest.fixture
def valid_dataframe():
    """Cria um conjunto mínimo de dados válidos."""

    return pd.DataFrame({
        "SeriousDlqin2yrs": [0, 1, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.30, 0.80, 0.15],
        "age": [35, 45, 60],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 0],
        "DebtRatio": [0.40, 0.70, 0.25],
        "MonthlyIncome": [5000.0, 3500.0, 7000.0],
        "NumberOfOpenCreditLinesAndLoans": [5, 8, 4],
        "NumberOfTimes90DaysLate": [0, 1, 0],
        "NumberRealEstateLoansOrLines": [1, 2, 1],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0],
        "NumberOfDependents": [1.0, 2.0, 0.0]
    })


def test_valid_dataframe_is_approved(valid_dataframe):
    """Dados que respeitam o contrato devem ser aprovados."""

    result = validate_credit_data(valid_dataframe)

    assert len(result) == 3
    assert list(result.columns) == list(valid_dataframe.columns)


def test_invalid_age_is_rejected(valid_dataframe):
    """Idade inferior a 18 deve reprovar o contrato."""

    valid_dataframe.loc[0, "age"] = 15

    with pytest.raises(SchemaError):
        validate_credit_data(valid_dataframe)


def test_negative_debt_ratio_is_rejected(valid_dataframe):
    """DebtRatio negativo deve reprovar o contrato."""

    valid_dataframe.loc[0, "DebtRatio"] = -0.5

    with pytest.raises(SchemaError):
        validate_credit_data(valid_dataframe)


def test_invalid_target_is_rejected(valid_dataframe):
    """Target diferente de 0 ou 1 deve reprovar o contrato."""

    valid_dataframe.loc[0, "SeriousDlqin2yrs"] = 2

    with pytest.raises(SchemaError):
        validate_credit_data(valid_dataframe)


def test_duplicate_records_are_rejected(valid_dataframe):
    """Duplicidades devem ser identificadas antes da validação."""

    duplicated_dataframe = pd.concat(
        [valid_dataframe, valid_dataframe.iloc[[0]]],
        ignore_index=True
    )

    with pytest.raises(ValueError, match="linhas duplicadas"):
        validate_credit_data(duplicated_dataframe)


def test_prepare_reference_data(valid_dataframe):
    """Preparação deve separar idade inválida e imputar ausentes."""

    dataframe = valid_dataframe.copy()

    dataframe.loc[0, "MonthlyIncome"] = None
    dataframe.loc[1, "NumberOfDependents"] = None
    dataframe.loc[2, "age"] = 15

    clean, invalid_age, duplicates = prepare_reference_data(dataframe)

    assert len(clean) == 2
    assert len(invalid_age) == 1
    assert len(duplicates) == 0

    assert clean["MonthlyIncome"].isna().sum() == 0
    assert clean["NumberOfDependents"].isna().sum() == 0

    validate_credit_data(clean, check_duplicates=False)
