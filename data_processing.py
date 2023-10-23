import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm


data = pd.read_csv('data/spotify_data.csv')


# Drop unnecessary columns
data = data.drop(['Unnamed: 0', 'artist_name', 'track_id', 'year'], axis=1)

# Categorical variables
columns_categorical = ['key', 'mode', 'time_signature', 'genre']
# Convert categorical variables to dummy variables (0 and 1)
data = pd.get_dummies(data, columns=columns_categorical, drop_first=True, dtype=int)

# transform popularity between 0 and 10: if popularity between 0 and 10, then 0, if between 11 and 20, then 1, etc.
def transform_popularity(popularity):
    if popularity <= 10:
        return 1
    elif popularity <= 20:
        return 2
    elif popularity <= 30:
        return 3
    elif popularity <= 40:
        return 4
    elif popularity <= 50:
        return 5
    elif popularity <= 60:
        return 6
    elif popularity <= 70:
        return 7
    elif popularity <= 80:
        return 8
    elif popularity <= 90:
        return 9
    elif popularity <= 100:
        return 10
    
# Apply the function to the popularity column
data['popularity'] = data['popularity'].apply(transform_popularity)

print(data.head())
    
# Save the data
data.to_csv('data/spotify_data_processed.csv', index=False)


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

def get_labels(data):
    ''' Get labels for each track based on embedding and clusters assignments '''
    # Compute embeddings
    embeddings = compute_embeddings(data)
    # Cluster embeddings
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    # Get labels
    labels = kmeans.labels_
    return labels

compute_embeddings(data['track_name'])

# Add labels to the data for tracks names
# data['labels'] = get_labels(data['track_name'])
# data = data.drop(['track_name'], axis=1)
    


def load_embeddings():
    ''' Load the embeddings from the pickle file '''
    with open('data/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

