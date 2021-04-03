# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
#from get_data import read_params
import argparse
import joblib
import json
import yaml
'''Class to train and evaluate.
Also write metrics to JSON file.
'''
class train_and_evaluate:

	def eval_metrics(self, actual, pred):
		rmse = np.sqrt(mean_squared_error(actual, pred))
		mae = mean_absolute_error(actual, pred)
		r2 = r2_score(actual, pred)
		return rmse, mae, r2

	def train_and_evaluate_func(self, config):
		#config = read_params(config_path)
		test_data_path = config["split_data"]["test_path"]
		train_data_path = config["split_data"]["train_path"]
		random_state = config["base"]["random_state"]
		model_dir = config["model_dir"]

		

		target = [config["base"]["target_col"]]
		print("target")
		train = pd.read_csv(train_data_path, sep=",")
		test = pd.read_csv(test_data_path, sep=",")

		y_train = train[target]
		y_test = test[target]

		X_train = train.drop(target, axis=1)
		X_test = test.drop(target, axis=1)
		
		sc = StandardScaler()
		sc.fit(X_train)
		X_train_scaled = sc.transform(X_train)
		X_test_scaled = sc.transform(X_test)
		
		pipe = Pipeline([
			('sc', StandardScaler()),
			('rf', RandomForestRegressor(random_state=42))
		])
		
		parameters = {
			'rf__n_estimators':[80,90,100]
		}
		
		gcv = GridSearchCV(pipe, parameters, cv=4, verbose=1)
		
		gcv.fit(X_train, y_train.values)
		
		predicted_qualities = gcv.predict(X_test)
		
		(rmse, mae, r2) = self.eval_metrics(y_test.values, predicted_qualities)

		#print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
		print("  RMSE: %s" % rmse)
		print("  MAE: %s" % mae)
		print("  R2: %s" % r2)

	#####################################################
		scores_file = config["reports"]["scores"]
		params_file = config["reports"]["params"]

		with open(scores_file, "w") as f:
			scores = {
				"rmse": rmse,
				"mae": mae,
				"r2": r2
			}
			json.dump(scores, f, indent=4)

		
	#####################################################


		os.makedirs(model_dir, exist_ok=True)
		model_path = os.path.join(model_dir, "model.joblib")

		joblib.dump(gcv, model_path)
		
	def __init__(self):
		self.raw_data = dict()
		self.path = 'params.yaml'
		with open(self.path) as f:
			self.raw_data  = yaml.safe_load(f)
		#print("as")
		self.train_and_evaluate_func(self.raw_data)
		

		
if __name__ == "__main__":
	train_and_evaluate()