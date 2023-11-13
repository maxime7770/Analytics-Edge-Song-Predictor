import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, mean_absolute_error


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# drop all columns starting with genre_
# train_data = train_data.drop(train_data.columns[train_data.columns.str.startswith('genre_')], axis=1)
# test_data = test_data.drop(test_data.columns[test_data.columns.str.startswith('genre_')], axis=1)


# change 1, 2, 3... to 0, 1, 2...
# train_data['popularity'] = train_data['popularity'] - 1
# test_data['popularity'] = test_data['popularity'] - 1

# # random subset of 50000 rows
# train_data = train_data.sample(n=100000, random_state=1)

print(train_data)

def model_xgboost(X_train, y_train):
    # use cross validation to select the best parameters
    parameters = {
        'max_depth': [5, 7, 10],
        'learning_rate': [0.05],
        'n_estimators': [100, 500],
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
    print('MAE', mean_absolute_error(test_data.iloc[:, 0], y_pred))

# model_xgboost(train_data.iloc[:, 1:], train_data.iloc[:, 0])
# select_xgboost()

def model_randomforest(X_train, y_train):
    # parameters = {
    #     'max_depth': [5, 10],
    #     'n_estimators': [100, 200],
    # }
    # rf = RandomForestClassifier()
    # clf = GridSearchCV(rf, parameters, cv=5, scoring='accuracy', verbose=1)
    clf = RandomForestClassifier(max_depth=10, n_estimators=100)
    clf.fit(X_train, y_train)
    return clf

def select_randomforest():
    best_rf = model_randomforest(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_rf, open('models_path/randomforest.sav', 'wb'))
    y_pred = best_rf.predict(test_data.iloc[:, 1:])

    print('Accuracy on train', accuracy_score(train_data.iloc[:, 0], best_rf.predict(train_data.iloc[:, 1:])))
    print('Accuracy on test: ', accuracy_score(test_data.iloc[:, 0], y_pred))
    print('MAE', mean_absolute_error(test_data.iloc[:, 0], y_pred))


def model_adaboost(X_train, y_train):
    # chose some parameters for cross-validation
    parameters = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.5],
    }
    ada = AdaBoostClassifier()
    clf = GridSearchCV(ada, parameters, cv=5, scoring='accuracy', verbose=1)
    clf.fit(X_train, y_train)
    return clf

def select_adaboost():
    best_ada = model_adaboost(train_data.iloc[:, 1:], train_data.iloc[:, 0])
    # save the model with pickle
    pickle.dump(best_ada, open('models_path/adaboost.sav', 'wb'))
    y_pred = best_ada.predict(test_data.iloc[:, 1:])

    print('Accuracy on train', accuracy_score(train_data.iloc[:, 0], best_ada.predict(train_data.iloc[:, 1:])))
    print('Accuracy on test: ', accuracy_score(test_data.iloc[:, 0], y_pred))
    print('MAE', mean_absolute_error(test_data.iloc[:, 0], y_pred))

# model_adaboost(train_data.iloc[:, 1:], train_data.iloc[:, 0])
# select_adaboost()


# model_randomforest(train_data.iloc[:, 1:], train_data.iloc[:, 0])
# select_randomforest()


rf = pickle.load(open('models_path/xgboost.sav', 'rb'))
# prediction on the whole dataset
y_pred = rf.predict(train_data.iloc[:, 1:])
# summary of the predictions (distribution)
print(pd.Series(y_pred).value_counts())