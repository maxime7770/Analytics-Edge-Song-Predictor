import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


data = pd.read_csv('data/spotify_data_processed.csv')



def compute_embeddings(data):
    ''' Given a list of strings (track names), get a list of embeddings '''
    embeddings = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for sentence in tqdm(data):
        embeddings.append(model.encode(sentence))

    # store the embeddings along with the track names
    embeddings = np.array(embeddings)
    # store in a pickle file
    with open('data/embeddings_reduced.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings



if __name__ == '__main__':
    compute_embeddings(data['track_name'])