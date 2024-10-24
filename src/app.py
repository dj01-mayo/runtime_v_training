import os
import sys
import json
import logging
import dateutil
import dateutil.parser
import pickle
import numpy as np
from numpy import ndarray
from typing import Literal
from datetime import datetime
from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import HTMLResponse,JSONResponse, Response
from fastapi.logger import logger
from pydantic import BaseModel,Field
from utilities import data_transformer, model_input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log_handl = logging.StreamHandler()
log.addHandler(hdlr=log_handl)

model_app = FastAPI()

base_path = os.path.abspath(os.path.curdir)
log.info(f'Loading data: {base_path}')

@model_app.get('/')
def read_root()-> list:
    """
    A basic endpoint returning Environment Variables to 
    verify that the FastAPI app is running.
    """
    return [f'{k}: {v}' for k, v in sorted(os.environ.items())]

@model_app.post('/predict')
def run_prediction(request: Request, input_values: model_input)-> JSONResponse:
    """
    The app's predict function. It takes a request and translates
    the request' values into a form the model can use. 

    Ultimately, the function returns a response with the prediction result.
    """
    if input_values is not None:
        log.info(f'Request received. Input Type: {type(input_values)},  Model inputs: {input_values}')
        with open(os.path.join(base_path,'data','encoder.pkl'),'rb') as enc_file:
            encoder = pickle.load(enc_file)
        input_dataframe = pd.DataFrame(data=input_values.model_dump(), index=range(0,1))
        
        # encoder = OneHotEncoder()
        transformed_values = encoder.transform(input_dataframe)
        # transformer = encoder.transform(pd.DataFrame(input_values.model_dump))
        # transformed_values = transformer.transform_data_model(data_to_transform=input_values)
        log.info(f'Input data encoded. Input Type: {type(input_values)},  Encoded Data: {transformed_values}')
    else:
        return HTTPException(status_code=400,detail=f'Request received; but model inputs malformed or absent.')
    
    with open(os.path.join(base_path,'data','model.pkl'),'rb') as mdl:
        model = pickle.load(mdl)
    log.info(f'Loaded Model.')
    try:
        # log.info(f'Input dataframe: {pd.DataFrame(transformed_values.model_dump(),index=range(0,1)).to_dict()}')
        prediction = model.predict(transformed_values)
        pickle_bytes = prediction.tolist()
        
        log.info(f'Prediction complete: Shape - {prediction.shape}, Prediction - {prediction[:1]}')
        return JSONResponse(content=json.dumps(pickle_bytes), status_code=200)
    except Exception as e:
        log.error(f'An error occurred during prediction. Error: {e}')
        return HTTPException(status_code=500, detail=f'An error occurred during prediction. Error: {e}')