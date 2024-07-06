import pytest

import sys, os
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT).replace('prediction_model', ''))
print(PACKAGE_ROOT)

from prediction_model.config import config
from prediction_model.predict import generate_prediction
from prediction_model.processing.data_handling import load_dataset


@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_PATH)
    single_row = test_data[:1]
    result = generate_prediction(single_row)
    
    return result


def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None
    
def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0], str)
    
def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0]=='Y'