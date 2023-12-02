import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from tqdm import tqdm


data = pd.read_csv('data/spotify_data.csv')


# Drop unnecessary columns
data = data.drop(['Unnamed: 0', 'artist_name', 'track_id', 'year'], axis=1)
data = data.dropna()
data = data.reset_index(drop=True)

print("Track names count: ", len(data['track_name']))

# Categorical variables
columns_categorical = ['key', 'mode', 'time_signature', 'genre']
# # Convert categorical variables to dummy variables (0 and 1)
data = pd.get_dummies(data, columns=columns_categorical, drop_first=True, dtype=int)


data = data[(data['popularity'] <= 50)]

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
# for each popularity level, select at most 20k rows
data = data.groupby('popularity').head(5000)

data = data.drop(['track_name'], axis=1)
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=0)
train.to_csv('data/train_regression.csv', index=False)
test.to_csv('data/test_regression.csv', index=False)
