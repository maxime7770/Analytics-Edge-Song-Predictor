import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


data = pd.read_csv('data/spotify_data_processed_clustering.csv')



def compute_embeddings(data):
    ''' Given a list of strings (track names), get a list of embeddings '''
    embeddings = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for sentence in tqdm(data):
        embeddings.append(model.encode(sentence))

    # store the embeddings along with the track names
    embeddings = np.array(embeddings)
    # store in a pickle file
    with open('data/embeddings_clustering.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings



if __name__ == '__main__':

    embeddings = pickle.load(open('data/embeddings_clustering.pkl', 'rb'))
    list_embeddings = list(embeddings)
    print('Embeddings loaded')
    print(len(list_embeddings))
    # Cluster embeddings
    kmeans = KMeans(n_clusters=10, random_state=0, verbose=1).fit(list_embeddings)
    # Get labels
    labels = kmeans.labels_
    data['track_name_labels'] = labels    
    data.to_csv('data/spotify_data_processed_clustering.csv', index=False)

    # scores = []
    # for k in [3, 5, 10, 15, 20]:
    #     kmeans = KMeans(n_clusters=k, random_state=0, verbose=1).fit(list_embeddings)
    #     # Get labels
    #     labels = kmeans.labels_
    #     scores.append(silhouette_score(list_embeddings, labels))
    # plt.plot([3, 5, 10, 15, 20], scores)