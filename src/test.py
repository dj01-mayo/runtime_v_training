import os
import sys
import pytest
import logging
import pickle
import json
import numpy
from numpy import ndarray, array
from requests import Request

#Set up import path
module_path = os.path.abspath(os.path.curdir)
sys.path.append(os.path.join(module_path,'src'))
print(f'Sys path: {sys.path}')
import app
from src.utilities import model_input, data_transformer

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def test_verify_base():
    result = app.read_root()
    assert (result is not None and
            type(result) == list)

def test_model_input_pydantic():
    mock_request = {'current_status':'Laboratory-confirmed case',
                    'sex':'Male',
                    'age_group':'0 - 9 Years',
                    'race_ethnicity_combined':'Black, Non-Hispanic',
                    'hosp_yn':'Yes',
                    'icu_yn': 'No',
                    'medcond_yn':'Yes'
                    }
    app_input = model_input(**mock_request)
    assert (app_input is not None and
            app_input.current_status == 'Laboratory-confirmed case')

def test_model_input_value_defaults():
    mock_request = {'current_status':'Laboratory-confirmed case',
                    'age_group':'0 - 9 Years',
                    }
    app_input = model_input(**mock_request)
    assert (app_input is not None and
            app_input.sex == 'Missing' and
            app_input.hosp_yn == 'Missing' and
            app_input.icu_yn == 'Missing' and
            app_input.medcond_yn == 'Missing' and
            app_input.race_ethnicity_combined == 'Missing')

def test_runtime_data_transformation():
    mock_request = {'current_status':'Laboratory-confirmed case',
                    'sex':'Male',
                    'age_group':'0 - 9 Years',
                    'race_ethnicity_combined':'Black, Non-Hispanic',
                    'hosp_yn':'Yes',
                    'icu_yn': 'No',
                    'medcond_yn':'Yes'
                    }
    app_input = model_input(**mock_request)
    transformer = data_transformer(log=log)
    transformed_model = transformer.transform_data_model(data_to_transform=app_input)
    assert (transformed_model is not None and
            transformed_model.sex != 'Male' and
            transformed_model.hosp_yn == 1 and
            transformed_model.icu_yn == 0 and
            transformed_model.medcond_yn == 1 and
            transformed_model.race_ethnicity_combined != 'Missing')

def test_model_prediction():
    mock_request = {'current_status':'Laboratory-confirmed case',
                    'sex':'Male',
                    'age_group':'0 - 9 Years',
                    'race_ethnicity_combined':'Black, Non-Hispanic',
                    'hosp_yn':'Yes',
                    'icu_yn': 'No',
                    'medcond_yn':'No'
                    }
    req = Request(method='post',data=mock_request)
    prediction = app.run_prediction(req, input_values=model_input(**mock_request))
    prediction_deserialize = json.loads(prediction.body)
    # results_array = array(prediction_deserialize)
    # results_array = numpy.fromstring(prediction_deserialize)
    log.info(f'test_model_input_value_defaults :: StatusCode: {prediction.status_code} Prediction - {prediction}')
    log.info(f'test_model_input_value_defaults :: Prediction Deserialized - {prediction_deserialize}')
    # log.info(f'test_model_input_value_defaults :: Prediction NDArray Loaded - {results_array}')
    assert prediction is not None