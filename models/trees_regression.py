import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pickle
import lightgbm as lgb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, mean_absolute_error


train_data = pd.read_csv('data/train_regression.csv')
test_data = pd.read_csv('data/test_regression.csv')


def model_xgboost_regressor(X_train, y_train):
    # use cross validation to select the best parameters
    parameters = {
        'max_depth': [5, 7, 10],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [50, 100, 200, 500],
    }
    xgb = XGBRegressor()
    clf = GridSearchCV(xgb, parameters, cv=5, scoring='neg_mean_absolute_error', verbose=1)
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def select_xgboost_regressor():
    best_xgb = model_xgboost_regressor(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_xgb, open('models_path/xgboost_regressor.sav', 'wb'))
    y_pred = best_xgb.predict(test_data.iloc[:, 1:])

    print('MAE', mean_absolute_error(test_data.iloc[:, 0], y_pred))
    print('R2 on train: ', best_xgb.score(train_data.iloc[:, 1:], train_data.iloc[:, 0]))
    print('R2 on test: ', best_xgb.score(test_data.iloc[:, 1:], test_data.iloc[:, 0]))

# model_xgboost_regressor(train_data.iloc[:, 1:], train_data.iloc[:, 0])
# select_xgboost_regressor()



def model_rf_regressor(X_train, y_train):
    parameters = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
    }
    clf = RandomForestRegressor()
    clf = GridSearchCV(clf, parameters, cv=5, scoring='neg_mean_absolute_error', verbose=1)
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def select_rf_regressor():
    best_rf = model_rf_regressor(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_rf, open('models_path/rf_regressor.sav', 'wb'))
    y_pred = best_rf.predict(test_data.iloc[:, 1:])
    print('MAE', mean_absolute_error(test_data.iloc[:, 0], y_pred))
    print('R2 on train: ', best_rf.score(train_data.iloc[:, 1:], train_data.iloc[:, 0]))
    print('R2 on test: ', best_rf.score(test_data.iloc[:, 1:], test_data.iloc[:, 0]))

# model_rf_regressor(train_data.iloc[:, 1:], train_data.iloc[:, 0])
# select_rf_regressor()

def model_adaboost_regressor(X_train, y_train):
    parameters = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.5],
    }
    ada = AdaBoostRegressor()
    clf = GridSearchCV(ada, parameters, cv=5, scoring='neg_mean_absolute_error', verbose=1)
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def select_adaboost_regressor():
    best_ada = model_adaboost_regressor(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_ada, open('models_path/adaboost_regressor.sav', 'wb'))
    y_pred = best_ada.predict(test_data.iloc[:, 1:])
    print('MAE', mean_absolute_error(test_data.iloc[:, 0], y_pred))
    print('R2 on train: ', best_ada.score(train_data.iloc[:, 1:], train_data.iloc[:, 0]))
    print('R2 on test: ', best_ada.score(test_data.iloc[:, 1:], test_data.iloc[:, 0]))

model_adaboost_regressor(train_data.iloc[:, 1:], train_data.iloc[:, 0])
select_adaboost_regressor()
