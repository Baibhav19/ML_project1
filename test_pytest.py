import json
import logging
import os
import joblib
import pytest
import yaml
import pandas as pd
from src.prediction import prediction
#from src import prediction

@pytest.fixture
def initialize_config():
    raw_data = dict()
    path = 'params.yaml'
    with open(path) as f:
        raw_data  = yaml.safe_load(f)
    return raw_data

def test_incorrect_range(initialize_config):
    print(initialize_config)
    #assert 4 == initialize_config
    random_state = initialize_config["base"]["random_state"]
    model_dir = initialize_config["model_dir"]
    model_path = os.path.join(model_dir, "model.joblib")
        
    columns = initialize_config["columns"]
    test_data_path = initialize_config["pytest_data"]["incorrect_range"]
    test_result_path = initialize_config["pytest_data"]["test_results"]
    schema_path = initialize_config["schema"]
    pred = prediction(type="pytest")
    data = pd.read_csv('test_data.csv')
    assert 'Values not in range' == pred.return_result()

