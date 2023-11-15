from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.utils import get_first_cabin, get_title


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""

    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.numerical_variables
        + config.model_config.categorical_variables
        and validated_data[var].isnull().sum() > 0
    ]

    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """check model inputs for unprocessable values."""

    # replace interrogation marks by NaN values
    input_data = input_data.replace("?", np.nan)

    # retain only the first cabin if more than
    # 1 are available per passenger
    input_data["cabin"] = input_data["cabin"].apply(get_first_cabin)

    # extracts the title (Mr, Ms, etc) from the name variable
    input_data["title"] = input_data["name"].apply(get_title)

    # cast numerical variables as floats
    input_data["fare"] = input_data["fare"].astype("float")
    input_data["age"] = input_data["age"].astype("float")

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    survived: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[str]
    body: Optional[str]
    homedest: Optional[str]
    title: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
