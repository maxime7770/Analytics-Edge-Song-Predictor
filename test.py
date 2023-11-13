import pandas as pd
import pickle

test = pd.read_csv('data/test.csv')
model_knn = pickle.load(open('models_path/knn_bestk_new.sav', 'rb'))

# transform test to a matrix
test = test.iloc[:, 1:].values
# distribution of predictions
print(pd.Series(model_knn.predict(test)).value_counts())