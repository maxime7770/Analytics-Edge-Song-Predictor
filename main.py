import streamlit as st
import random
import pickle
from utils import *


xgb_model = pickle.load(open('models_path/xgboost.sav', 'rb'))

st.set_page_config(layout="wide")
st.title("Song Popularity Predictor")
st.markdown("**Give your song's characteristics and we'll predict its popularity!**")

title = st.text_input("Song Title", "")
liveliness = st.slider("Liveliness (1-100)", 0, 100, 50, 5)
expressivity = st.slider("Expressivity (1-100)", 0, 100, 50, 5)
instrumentalness = st.slider("Instrumentalness (1-100)", 0, 100, 50, 5)
tempo = st.slider("Tempo (1-100)", 1, 100, 50)

predicted_popularity = xgb_model.predict([[liveliness, expressivity, instrumentalness, tempo]])[0] + 1

st.header("Predictions")

animation(predicted_streams, "Predicted Number of Streams", COLOR_BLUE)
animation(predicted_popularity, "Predicted Popularity", COLOR_CYAN)


st.header("Similar Songs")
similar_songs = ["Song A", "Song B", "Song C", "Song D", "Song E"]
st.write(similar_songs)
