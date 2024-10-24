import os
import sys
import json
import logging
import dateutil
import dateutil.parser
import pickle
import warnings
import pandas as pd
import numpy
from utilities import data_transformer
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set up logging.
# This can be much simpler to use than print() statements.

# Notice the use of getLogger() ... This helps you hook into
# the existing logging python is running.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log_handl = logging.StreamHandler()
log.addHandler(hdlr=log_handl)

# Short-cut code to let us avoid having to paste in 
# a file path everytime we run this script somewhere new.
base_path = os.path.abspath(os.path.curdir)
log.info(f'Loading data: {base_path}')
data = pd.read_csv(os.path.join(base_path, 'data', 'testdata_COVID-19_Case_Surveillance_Public_Use_Data_May_2021.csv'), parse_dates=True)
original_data = data
enc = LabelEncoder()
cols_to_transform = ['current_status',
                     'sex',
                     'age_group',
                     'race_ethnicity_combined']

# This function just provides a basic transformation
# as several of the yes/no columns in the data have other values too. 
def class_encoding(val):
    if val in ["Missing","Unknown"]:
        return 3
    elif val == "No":
        return 0
    else:
        return 1
    
log.info(f'Encoding multiclass-Y/N columns.')

# Encode the y/n multi-class columns as a default encoder doesn't handle this well.
# Map() is a shortcut to perform the same action on item in a list or dataframe.
# Since we're identifying a single column, the "items" end up being each row in the column.

# data['medcond_yn'] = data['medcond_yn'].map(lambda x: class_encoding(x))
# data['death_yn'] = data['death_yn'].map(lambda x: class_encoding(x))
# data['icu_yn'] = data['icu_yn'].map(lambda x: class_encoding(x))
# data['hosp_yn'] = data['hosp_yn'].map(lambda x: class_encoding(x))
# data['cdc_case_earliest_dt'] = data['cdc_case_earliest_dt '].map(lambda x: datetime.strptime(x, '%Y/%m/%d').toordinal())
# log.info(f'Encoding remaining columns.')
# for col in cols_to_transform:
#     data[col] = enc.fit_transform(data[col])
# transformer = data_transformer(log=log)
# encoded_data = transformer.transform_dataframe(data=data)
# Omit columns which are not input values.

X = data.drop(columns=['death_yn','cdc_case_earliest_dt ','cdc_report_dt','onset_dt','pos_spec_dt'])
# dict_vect = DictVectorizer()
onehot = OneHotEncoder()
X_Encoded = onehot.fit_transform(X=X)
# clean_data = data.drop(columns=['death_yn','cdc_case_earliest_dt ','cdc_report_dt','onset_dt','pos_spec_dt'])
# X = dict_vect.fit_transform([clean_data.to_dict()])
log.info(f'Training data sample: {X.head(n=10)}')
# Limit the columns to only the one we want to see the prediction in.
Y = data['death_yn']

# Use scikit learn's native dataset splitting capability.
log.info(f'Splitting Data.')
X_train, X_test, y_train, y_test = train_test_split(X_Encoded, Y, test_size=0.2, random_state=42)

# Train the model.
# NOTE: Much work can be done here to improve this model.
# This content is for illustrative purposes only.
log.info(f'Running training.')
model = LogisticRegression(verbose=1)
model.fit(X_train, y_train)

# Run a prediction so that we can get some metrics.
log.info(f'Running prediction.')
y_pred = model.predict(X_test)

# One of many possible metrics to use. 
# Accuracy is used here as a simple proxy for illustrative purposes.
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100
# If accuracy is higher than 50%, save the model inputs as a artifact;
# Else, raise a warning. 
if accuracy_percentage > 50:
    # This gives us an artifact we can use when we move to run-time.
    log.info(f'Accuracy: {accuracy_percentage:.2f}%')
    model_params = model.get_params(deep=True)
    model_bytes = pickle.dumps(model)
    with open(os.path.join(base_path,'data','model.pkl'), 'wb') as m:
        pickle.dump(obj=model, file=m)
        m.close()
    with open(os.path.join(base_path,'data','encoder.pkl'), 'wb') as f:
        pickle.dump(obj=onehot, file=f)
        f.close()
    # with open(os.path.join(base_path,'data','model_params.json'), 'w') as f:
    #     f.write(json.dumps(model_params))
    #     f.close()
else:
    warnings.warn(f'Model accuracy below threshold: Accuracy - {accuracy_percentage:.2f}')

