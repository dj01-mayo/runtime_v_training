import os
import sys
import json
import logging
import dateutil
import dateutil.parser
import pickle
import warnings
from datetime import datetime

import pandas as pd
from utilities import model_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class data_transformer():
    """
    A class to abstract data transformations out of the training and application code.

    This allows us to have consistent transformations during training and runtime.
    """
    def __init__(self, log: logging.Logger = None, 
                 scikit_encoder_cols:list[str]=None,
                 yn_multiclass_fields:list[str]=None,
                 date_time_fields:list[str]=None) -> None:
        self.log = log
        self.scikit_encoder_cols = scikit_encoder_cols if scikit_encoder_cols is not None else ['current_status',
                                                                                                'sex',
                                                                                                'age_group',
                                                                                                'race_ethnicity_combined']
        self.yn_multiclass_fields = yn_multiclass_fields if yn_multiclass_fields is not None else ['medcond_yn',
                                                                                                   'death_yn',
                                                                                                   'icu_yn',
                                                                                                   'hosp_yn']
        self.date_time_fields = date_time_fields if date_time_fields is not None else ['cdc_case_earliest_dt ']

    def _class_encoding(self, val: str) -> int:
            """
            Private function to assist encoding yes/no fields. 

            For values outside of 'yes' or 'no', 3 is the encoded value.
            """
            if val.lower() == 'yes':
                return 1
            elif val.lower() == "no":
                return 0
            else:
                return 3
    def transform_data_model(self, data_to_transform: model_input):
        enc = LabelEncoder()
        for k in data_to_transform.model_fields.keys():
            if k in self.scikit_encoder_cols:
                data_to_transform[k] = enc.fit_transform([data_to_transform[k]])[0]
            elif k in self.yn_multiclass_fields:
                data_to_transform[k] = self._class_encoding(data_to_transform[k])
            else:
                data_to_transform[k] = datetime.strptime(data_to_transform[k], '%Y/%m/%d').toordinal()
        self.log.info(f'Encoding multiclass-Y/N columns.')
        return data_to_transform
    
    def transform_dataframe(self, data: pd.DataFrame):
        enc = LabelEncoder()
        # Encode the y/n multi-class columns as a default encoder doesn't handle this well.
        # Map() is a shortcut to perform the same action on item in a list or dataframe.
        # Since we're identifying a single column, the "items" end up being each row in the column.
        for yn_col in self.yn_multiclass_fields:
            data[yn_col] = data[yn_col].map(lambda x: self._class_encoding(x))
        for dtm_col in self.date_time_fields:
            data[dtm_col] = data[dtm_col].map(lambda x: datetime.strptime(x, '%Y/%m/%d').toordinal())
        self.log.info(f'Encoding remaining columns.')
        for col in self.scikit_encoder_cols:
            data[col] = enc.fit_transform(data[col])
        return data