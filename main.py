import streamlit as st
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import *
from inference import inference_xgboost, inference_random_forest, inference_adaboost, ensemble_prediction, inference_reg_lasso, inference_reg_ridge, inference_knn, ensemble_model
import shap
import pickle 
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


def transform_format(x, list_columns):
    ''' transform format of a new sample x to match the columns of training data '''
    key = x['key'].values[0]
    time_signature = x['time_signature'].values[0]
    genre = x['genre'].values[0]
    # values of 1 in the new columns
    x[f'key_{key}'] = 1
    x[f'time_signature_{time_signature}'] = 1
    x[f'genre_{genre}'] = 1
    # drop old columns
    x = x.drop(['key', 'time_signature', 'genre'], axis=1)

    x = x.reindex(columns=list_columns, fill_value=0)

    return x.drop(['popularity'], axis=1)

st.set_page_config(layout="wide")
st.title("Song Popularity Predictor")
st.markdown("**Give your song's characteristics and we'll predict its popularity!**")

def load_data():
    ''' load data from csv file '''
    train = pd.read_csv('data/train.csv')
    return train

train = load_data()

scaler = StandardScaler()
scaler.fit(train.iloc[:, 1:])

title = st.text_input("Song Title", "")
danceability = st.slider("Danceability (1-100)", 0, 100, 50, 5) / 100
energy = st.slider("Energy (1-100)", 0, 100, 50, 5) / 100
loudness = st.slider("Loudness (-50, 0)", -50, 0, -25)
speechiness = st.slider("Speechiness (1-100)", 0, 100, 50, 5) / 100
acousticness = st.slider("Acousticness (1-100)", 0, 100, 50, 5) / 100
instrumentalness = st.slider("Instrumentalness (1-100)", 0, 100, 50, 5) / 100
liveliness = st.slider("Liveliness (1-100)", 0, 100, 50, 5) / 100
valence = st.slider("Valence (1-100)", 0, 100, 50, 5) / 100
tempo = st.slider("Tempo (1-240)", 1, 240, 50)
duration = st.slider("Duration (in seconds)", 1, 1000, 200) * 1000

key = st.selectbox("Key", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
time_signature = st.selectbox("Time Signature", [1, 2, 3, 4, 5])



# choose the genre among different options
# options are names of columns that start with 'genre_'
options = [col for col in train.columns if col.startswith("genre_")]

# Create a list without the "genre_" prefix
options = [option.replace("genre_", "") for option in options]

# let the user choose the genre
genre = st.selectbox("Genre", options)

# danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo
sample = pd.DataFrame([[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveliness, valence, tempo, duration, key, time_signature, genre]], columns=['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'time_signature', 'genre'])
sample = transform_format(sample, train.columns)

print(sample)
sample.to_csv('data/sample.csv')

# add a button to run predictions
if st.button("Predict"):

    # predicted_popularity = random.randint(1, 10)
    predicted_popularity_xgb = inference_xgboost('models_path/xgboost.sav', sample)
    predict_popularity_rf = inference_random_forest('models_path/randomforest.sav', sample)
    predict_popularity_ada = inference_adaboost('models_path/adaboost.sav', sample)
    predicted_popularity_lasso = inference_reg_lasso('models_path/log_reg_lasso_new.sav', sample, scaler)
    predicted_popularity_ridge = inference_reg_ridge('models_path/log_reg_ridge_new.sav', sample, scaler)
    predicted_popularity_knn = inference_knn('models_path/knn_bestk_new.sav', sample, scaler)

    predicted_popularity = ensemble_prediction([predicted_popularity_xgb, predict_popularity_rf, predict_popularity_ada, predicted_popularity_lasso, predicted_popularity_ridge, predicted_popularity_knn])

    print(predict_popularity_ada, predicted_popularity_xgb, predict_popularity_rf, predicted_popularity_lasso, predicted_popularity)

    st.header("Predictions")

    animation(predicted_popularity, "Predicted Popularity", COLOR_CYAN)

    # panels with detail for each model
    with st.expander("See details for each model"):
        #bullet point for each model
        st.markdown(f"""
        - XGBoost: {predicted_popularity_xgb}
        - Random Forest: {predict_popularity_rf}
        - Adaboost: {predict_popularity_ada}
        - Lasso Regression: {predicted_popularity_lasso}
        - Ridge Regression: {predicted_popularity_ridge}
        - KNN: {predicted_popularity_knn}
        """)

    with st.expander("Feature importance"):
        model = pickle.load(open('models_path/xgboost.sav', 'rb'))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # fig = shap.force_plot(explainer.expected_value[predicted_popularity_xgb], shap_values[predicted_popularity_xgb], sample, matplotlib=True)
        # st.pyplot(fig, bbox_inches='tight')

        # abs_shap_values = abs(shap_values[predicted_popularity_xgb])
        # fig, ax = plt.subplots()
        # shap.summary_plot(abs_shap_values, sample, plot_type="bar", show=False)
        # st.pyplot(fig, bbox_inches='tight')
        # fig, ax = plt.subplots()
        # shap.force_plot(explainer.expected_value[predicted_popularity_xgb], shap_values[predicted_popularity_xgb], sample, matplotlib=True)
        # st.pyplot(fig, bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(5, 4))  # Adjust size as needed
        # shap.summary_plot(shap_values[predicted_popularity_xgb], sample, plot_type="bar", show=False)
        # st.pyplot(fig)

        st.info('''
                - Blue features contribute to HIGH popularity (push the model's probability of predicting the observed outcome higher)
                - Red features contribute to LOW popularity.
                - f(x) is the XGBoost model's "margin" score for the sample x.''')
        # For the SHAP force plot
        shap.force_plot(explainer.expected_value[predicted_popularity_xgb-1], shap_values[predicted_popularity_xgb-1], sample, matplotlib=True)
        st.pyplot(bbox_inches='tight')


        # def shap_model(x):
        #     return ensemble_model(x, scaler)
        
        # explainer = shap.KernelExplainer(shap_model)
        # shap_values = explainer.shap_values(sample)
        # print(shap_values)
        # st.pyplot(shap.force_plot(explainer.expected_value, shap_values, sample, matplotlib=True), bbox_inches='tight')


    

st.header("Similar Songs")
st.info("Coming soon!")
similar_songs = ["Song A", "Song B", "Song C", "Song D", "Song E"]
st.write(similar_songs)
