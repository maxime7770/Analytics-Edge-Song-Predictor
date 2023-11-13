import pickle
import numpy as np
import pandas as pd
import xgboost as xgb 


def inference_xgboost(path_model, sample):
    ''' Given a new sample, predict its popularity using XGBoost '''
    # load the model
    model = pickle.load(open(path_model, 'rb'))
    # predict the popularity
    prediction = model.predict(sample) + 1
    return prediction[0]

def inference_random_forest(path_model, sample):
    ''' Given a new sample, predict its popularity using Random Forest '''
    # load the model
    model = pickle.load(open(path_model, 'rb'))
    # predict the popularity
    prediction = model.predict(sample)
    return prediction[0]

def inference_adaboost(path_model, sample):
    ''' Given a new sample, predict its popularity using Adaboost '''
    # load the model
    model = pickle.load(open(path_model, 'rb'))
    # predict the popularity
    prediction = model.predict(sample)
    return prediction[0]

def inference_reg_lasso(path_model, sample, scaler):
    ''' Given a new sample, predict its popularity using Lasso '''
    # load the model
    model = pickle.load(open(path_model, 'rb'))
    # predict the popularity
    sample = scaler.transform(sample)
    prediction = model.predict(sample)
    return prediction[0]

def inference_reg_ridge(path_model, sample, scaler):
    ''' Given a new sample, predict its popularity using Ridge '''
    # load the model
    model = pickle.load(open(path_model, 'rb'))
    # predict the popularity
    sample = scaler.transform(sample)
    prediction = model.predict(sample)
    return prediction[0]

def inference_knn(path_model, sample, scaler):
    ''' Given a new sample, predict its popularity using kNN '''
    # load the model
    model = pickle.load(open(path_model, 'rb'))
    # predict the popularity
    sample = scaler.transform(sample)
    prediction = model.predict(sample)
    return prediction[0]




def ensemble_prediction(list_preds):
    ''' Given a list of predictions (labels between 1 and 10), compute the final prediction '''
    # most frequent prediction
    prediction = max(set(list_preds), key=list_preds.count)
    # if counts if only 1, then take the mean instead and take the closest integer
    if list_preds.count(prediction) == 1:
        prediction = round(np.mean(list_preds))
    return prediction



def ensemble_model(sample, scaler):
    ''' Given a new sample, predict its popularity using all the models '''
    # load the models
    path_model_xgboost = 'models_path/xgboost.sav'
    path_model_randomforest = 'models_path/randomforest.sav'
    path_model_adaboost = 'models_path/adaboost.sav'
    path_model_lasso = 'models_path/log_reg_lasso_new.sav'
    path_model_ridge = 'models_path/log_reg_ridge_new.sav'
    path_model_knn = 'models_path/knn_bestk_new.sav'
    # predict the popularity
    prediction_xgboost = inference_xgboost(path_model_xgboost, sample)
    prediction_randomforest = inference_random_forest(path_model_randomforest, sample)
    prediction_adaboost = inference_adaboost(path_model_adaboost, sample)
    prediction_lasso = inference_reg_lasso(path_model_lasso, sample, scaler)
    prediction_ridge = inference_reg_ridge(path_model_ridge, sample, scaler)
    prediction_knn = inference_knn(path_model_knn, sample, scaler)
    # ensemble the predictions
    prediction = ensemble_prediction([prediction_xgboost, prediction_randomforest, prediction_adaboost, prediction_lasso, prediction_ridge, prediction_knn])
    return prediction
