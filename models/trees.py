import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


train_data = pd.read_csv('data/train.csv').drop(['track_name_labels'], axis=1)
test_data = pd.read_csv('data/test.csv').drop(['track_name_labels'], axis=1)

# drop all columns starting with genre_
# train_data = train_data.drop(train_data.columns[train_data.columns.str.startswith('genre_')], axis=1)
# test_data = test_data.drop(test_data.columns[test_data.columns.str.startswith('genre_')], axis=1)


# change 1, 2, 3... to 0, 1, 2...
train_data['popularity'] = train_data['popularity'] - 1
test_data['popularity'] = test_data['popularity'] - 1

# random subset of 50000 rows
train_data = train_data.sample(n=50000, random_state=1)

print(train_data.head())

def model_xgboost(X_train, y_train):
    # use cross validation to select the best parameters
    parameters = {
        'max_depth': [5],
        'learning_rate': [0.05],
        'n_estimators': [500]
    }
    xgb = XGBClassifier()
    clf = GridSearchCV(xgb, parameters, cv=5, scoring='accuracy', verbose=1)
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def select_xgboost():
    best_xgb = model_xgboost(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_xgb, open('models_path/xgboost.sav', 'wb'))
    y_pred = best_xgb.predict(test_data.iloc[:, 1:])

    print('Accuracy on train: ', accuracy_score(train_data.iloc[:, 0], best_xgb.predict(train_data.iloc[:, 1:])))
    print('Accuracy on test: ', accuracy_score(test_data.iloc[:, 0], y_pred))

    

def model_randomforest(X_train, y_train):
    # parameters = {
    #     'max_depth': [5, 10],
    #     'n_estimators': [100, 200],
    # }
    # rf = RandomForestClassifier()
    # clf = GridSearchCV(rf, parameters, cv=5, scoring='accuracy', verbose=1)
    clf = RandomForestClassifier(max_depth=10, n_estimators=200)
    clf.fit(X_train, y_train)
    return clf

def select_randomforest():
    best_rf = model_randomforest(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_rf, open('models_path/randomforest.sav', 'wb'))
    y_pred = best_rf.predict(test_data.iloc[:, 1:])

    print('Accuracy: ', accuracy_score(test_data.iloc[:, 0], y_pred))


model_randomforest(train_data.iloc[:, 1:], train_data.iloc[:, 0])
select_randomforest()