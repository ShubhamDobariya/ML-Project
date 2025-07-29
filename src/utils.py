import os 
import sys

import numpy as np 
import pandas as pd
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
          dill.dump(obj, file_obj)  
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        best_models = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            print("Model : ", model)
            print("Model Name :", model_name)
            print("-----------------------------------------------------")
            
            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    verbose=0,
                    scoring='r2'
                )
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(x_train, y_train)
                best_model = model
            
            
            # model.fit(x_train, y_train) # Train model
            
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            best_models[model_name] = best_model
            
        return report, best_models
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)