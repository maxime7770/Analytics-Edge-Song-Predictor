import streamlit as st
import random

import pandas as pd
from utils import *


def transform_format(x, list_columns):
    ''' transform format of a new sample x to match the columns of training data '''
    key = new_sample_df['key'].values[0]
    time_signature = new_sample_df['time_signature'].values[0]
    genre = new_sample_df['genre'].values[0]
    # rename columns
    new_sample_df = new_sample_df.rename(columns={'key': f'key_{key}', 'time_signature': f'time_signature_{time_signature}', 'genre': f'genre_{genre}', 'mode': 'mode_1'})

    new_sample_df = x.reindex(columns=list_columns, fill_value=0)

    pass

st.set_page_config(layout="wide")
st.title("Song Popularity Predictor")
st.markdown("**Give your song's characteristics and we'll predict its popularity!**")

@st.cache
def load_data():
    ''' load data from csv file '''
    data = pd.read_csv('data/spotify_data_processed.csv')
    return data

data = load_data()

title = st.text_input("Song Title", "")
liveliness = st.slider("Liveliness (1-100)", 0, 100, 50, 5)
expressivity = st.slider("Expressivity (1-100)", 0, 100, 50, 5)
instrumentalness = st.slider("Instrumentalness (1-100)", 0, 100, 50, 5)
tempo = st.slider("Tempo (1-100)", 1, 100, 50)


# choose the genre among different options
# options are names of columns that start with 'genre_'
options = [col for col in data.columns if col.startswith("genre_")]

# Create a list without the "genre_" prefix
options = [option.replace("genre_", "") for option in options]

# let the user choose the genre
genre = st.selectbox("Genre", options)

predicted_streams = random.randint(1000, 1000000)
predicted_popularity = random.randint(1, 10)

st.header("Predictions")

animation(predicted_streams, "Predicted Number of Streams", COLOR_BLUE)
animation(predicted_popularity, "Predicted Popularity", COLOR_CYAN)


st.header("Similar Songs")
similar_songs = ["Song A", "Song B", "Song C", "Song D", "Song E"]
st.write(similar_songs)
