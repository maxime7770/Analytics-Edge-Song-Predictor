import pickle
import numpy as np
import pandas as pd
import xgboost as xgb 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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

def ensemble_prediction_regression(list_preds):
    ''' Given a list of predictions (labels between 1 and 10), compute the final prediction '''
    # most frequent prediction
    prediction = np.mean(list_preds)
    # round to closest integer
    prediction = round(prediction)
    prediction /= 5
    # if smaller than 0 or larger than 10, choose 0 or 10
    if prediction < 0:
        prediction = 0
    elif prediction > 10:
        prediction = 10
    # round again
    prediction = round(prediction)
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


def evaluate_ensemble():
    train_data = pd.read_csv('data/train.csv')
    scaler = StandardScaler()
    scaler.fit(train_data.iloc[:, 1:])
    test_data = pd.read_csv('data/test_regression.csv')
    y = test_data['popularity']
    test_data.drop("popularity", axis=1, inplace=True)
    scaled_test_data = scaler.transform(test_data)
    #     predicted_popularity_xgb = inference_xgboost('models_path_regression/xgboost_regressor.sav', sample)
    # predict_popularity_rf = inference_random_forest('models_path_regression/rf_regressor.sav', sample)
    # predict_popularity_ada = inference_adaboost('models_path_regression/adaboost_regressor.sav', sample)
    # predicted_popularity_linear = inference_reg_lasso('models_path_regression/linear.sav', sample, scaler)
    # predicted_popularity_glm_ridge = inference_reg_ridge('models_path_regression/ridgeGLM.sav', sample, scaler)
    # predicted_popularity_glm = inference_reg_ridge('models_path_regression/glm.sav', sample, scaler)
    # predicted_popularity_svm = inference_reg_ridge('models_path_regression/full_svm.pkl', sample, scaler)
    # predicted_popularity_knn = inference_knn('models_path_regression/knnreg_bestk23.sav', sample, scaler)
    model_xgb = pickle.load(open('models_path_regression/xgboost_regressor.sav', 'rb'))
    model_rf = pickle.load(open('models_path_regression/rf_regressor.sav', 'rb'))
    model_ada = pickle.load(open('models_path_regression/adaboost_regressor.sav', 'rb'))
    model_linear = pickle.load(open('models_path_regression/linear.sav', 'rb'))
    model_ridge = pickle.load(open('models_path_regression/ridgeGLM.sav', 'rb'))
    model_glm = pickle.load(open('models_path_regression/glm.sav', 'rb'))
    model_knn = pickle.load(open('models_path_regression/knnreg_bestk23.sav', 'rb'))
    pred_xgb = model_xgb.predict(test_data)
    pred_rf = model_rf.predict(test_data)
    pred_ada = model_ada.predict(test_data)
    pred_linear = model_linear.predict(scaled_test_data)
    pred_ridge = model_ridge.predict(scaled_test_data)
    pred_glm = model_glm.predict(scaled_test_data)
    pred_knn = model_knn.predict(scaled_test_data)

    predictions = []
    for i in range(len(pred_xgb)):
        predictions.append(np.mean([pred_xgb[i], pred_rf[i], pred_ada[i], pred_linear[i], pred_ridge[i], pred_glm[i], pred_knn[i]]))
    print('MAE', mean_absolute_error(y, predictions))
    print('R2', r2_score(y, predictions))

if __name__ == '__main__':
    evaluate_ensemble()