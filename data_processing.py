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

# transform popularity between 0 and 10: if popularity between 0 and 10, then 0, if between 11 and 20, then 1, etc.
def transform_popularity(popularity):
    if popularity <= 4:
        return 1
    elif popularity <= 8:
        return 2
    elif popularity <= 12:
        return 3
    elif popularity <= 16:
        return 4
    elif popularity <= 20:
        return 5
    elif popularity <= 24:
        return 6
    elif popularity <= 28:
        return 7
    elif popularity <= 32:
        return 8
    elif popularity <= 36:
        return 9
    else:
        return 10
    
# # Apply the function to the popularity column
# data['popularity'] = data['popularity'].apply(transform_popularity)


def compute_embeddings(data):
    ''' Given a list of strings (track names), get a list of embeddings '''
    embeddings = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for sentence in tqdm(data):
        embeddings.append(model.encode(sentence))

    # store the embeddings along with the track names
    embeddings = np.array(embeddings)
    dict_embeddings = {}
    for i in range(len(data)):
        dict_embeddings[data[i]] = embeddings[i]
    # store in a pickle file
    with open('data/embeddings.pkl', 'wb') as f:
        pickle.dump(dict_embeddings, f)
    
    return embeddings


def load_embeddings():
    ''' Load the embeddings from the pickle file '''
    with open('data/embeddings_raw.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def get_labels():
    ''' Get labels for each track based on embedding and clusters assignments '''
    # Compute embeddings from the dictionary
    list_embeddings = load_embeddings()
    print('Embeddings loaded')
    print(len(list_embeddings))
    # Cluster embeddings
    kmeans = KMeans(n_clusters=10, random_state=0, verbose=1).fit(list_embeddings)
    # Get labels
    labels = kmeans.labels_
    return labels

# Add labels to the data for tracks names
# data['track_name_labels'] = get_labels()


data = data[(data['popularity'] <= 40)]
# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
# for each popularity level, select at most 20k rows
data = data.groupby('popularity').head(20000)
# Create labels ranging from 1 to 10 based on the filtered 'popularity' values
data['popularity'] = data['popularity'].apply(transform_popularity)
data = data.groupby('popularity').head(12000)

# drop all rows where 'genre_songwriter' is equal to 1
# data = data[data['genre_songwriter'] == 0]
# # drop the 'genre_songwriter' column
# data = data.drop(['genre_songwriter'], axis=1)


# Save the data
data.to_csv('data/spotify_data_processed.csv', index=False)

data = data.drop(['track_name'], axis=1)


# split the data into train and test
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=0)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)