import os
import yaml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import json

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)

class prediction:
    
    def __init__(self, type="normal"):
        self.raw_data = dict()
        self.path = 'params.yaml'
        with open(self.path) as f:
            self.raw_data  = yaml.safe_load(f)
            #print("as")
        random_state = self.raw_data["base"]["random_state"]
        model_dir = self.raw_data["model_dir"]
        model_path = os.path.join(model_dir, "model.joblib")
        
        columns = self.raw_data["columns"]
        if (type == "pytest"):
            test_data_path = self.raw_data["pytest_data"]["incorrect_range"]
            test_result_path = self.raw_data["pytest_data"]["test_results"]
        else:
            test_data_path = self.raw_data["test_data"]["test_data_csv"]
            test_result_path = self.raw_data["test_data"]["test_results"]
        schema_path = self.raw_data["schema"]
        self.result = self.predict_func(test_data_path, test_result_path, model_path, columns, schema_path)
    
    def return_result(self):
        return self.result
    def predict_func(self, test_data_path, test_result_path, model_path, columns, schema_path):
        model = joblib.load(model_path)
        X_test = pd.read_csv(test_data_path)
        #print(X_test.drop('Unnamed: 0', axis=1).to_dict('list'))
        try:
            self.validate_input(X_test.drop('Unnamed: 0', axis=1).to_dict('list'), schema_path, columns)
        except NotInCols:
            return "Columns does not match"
        except NotInRange:
            return "Values not in range"
        pred = model.predict(X_test[columns])
        X_test['TARGET'] = pred
        #print(X_test)
        X_test['TARGET'] = X_test['TARGET'].apply( lambda x : round(x))
        #print(X_test)
        pd.DataFrame(X_test).to_csv(test_result_path)
        return "200 OK"

    def get_schema(self, schema_path):
        #print(schema_path)
        with open(schema_path) as json_file:
            schema = json.load(json_file)
        return schema

    def validate_input(self, dict_request, schema_path, columns):
        def _validate_cols(col):
            #schema = self.get_schema(schema_path = config["schema"])
            
            actual_cols = columns
            if col not in actual_cols:
                raise NotInCols

        def _validate_values(col, val):
            #print(schema_path)
            schema = self.get_schema(schema_path)
            #print(schema)
            #print(dict_request[col])
            for val in dict_request[col]:
                if not (schema["min"][col] <= float(val) <= schema["max"][col]):
                    raise NotInRange

        for col, val in dict_request.items():
            #print(col," hh ",val)
            _validate_cols(col)
            _validate_values(col, val)
    
        return True

if __name__ == "__main__":
    pred = prediction()