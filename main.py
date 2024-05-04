import streamlit as st
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import *
from inference import inference_xgboost, inference_random_forest, inference_adaboost, ensemble_prediction, inference_reg_lasso, inference_reg_ridge, inference_knn, ensemble_model, ensemble_prediction_regression
import shap
import pickle 
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go
import pickle
from sklearn.metrics import pairwise_distances
import seaborn as sns
import torch
from ensemble_stacking import Ensemble


st.set_option('deprecation.showPyplotGlobalUse', False)

# load nn model and scalers for ensemble
model = Ensemble(7, 5, 1)
model.load_state_dict(torch.load('models_path_regression/nn_ensemble.ckpt'))
model.eval()
scaler_ensemble = pickle.load(open('models_path_regression/scaler_ensemble.sav', 'rb'))
scaler_y = pickle.load(open('models_path_regression/scaler_ensemble_y.sav', 'rb'))



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

key = st.selectbox("Key", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help="The 'Key' of a song, with values between 1 and 10, indicates its musical scale, like how a '6' might represent the key of G major.")
time_signature = st.selectbox("Time Signature", [1, 2, 3, 4, 5], help="The 'Time Signature' of a song, with values between 1 and 5, indicates the number of beats per measure.")



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
    predicted_popularity_xgb = inference_xgboost('models_path_regression/xgboost_regressor.sav', sample)
    predict_popularity_rf = inference_random_forest('models_path_regression/rf_regressor.sav', sample)
    predict_popularity_ada = inference_adaboost('models_path_regression/adaboost_regressor.sav', sample)
    predicted_popularity_linear = inference_reg_lasso('models_path_regression/linear.sav', sample, scaler)
    predicted_popularity_glm_ridge = inference_reg_ridge('models_path_regression/ridgeGLM.sav', sample, scaler)
    predicted_popularity_glm = inference_reg_ridge('models_path_regression/glm.sav', sample, scaler)
    #predicted_popularity_svm = inference_reg_ridge('models_path_regression/full_svm.pkl', sample, scaler)
    predicted_popularity_knn = inference_knn('models_path_regression/knnreg_bestk23.sav', sample, scaler)

    predictions = [predicted_popularity_xgb, predict_popularity_rf, predict_popularity_ada, predicted_popularity_linear, predicted_popularity_glm_ridge, predicted_popularity_glm, predicted_popularity_knn]
    predictions = np.array(predictions)
    predictions = scaler_ensemble.transform(predictions.reshape(1, -1))
    pred_model = model(torch.tensor(predictions).float())
    pred_model = scaler_y.inverse_transform(pred_model.detach().numpy().reshape(-1, 1))
    pred_model = pred_model[0][0] / 5
    if pred_model < 0:
        pred_model = 0
    elif pred_model > 10:
        pred_model = 10
    predicted_popularity = round(pred_model)

    #print(predict_popularity_ada, predicted_popularity_xgb, predict_popularity_rf, predicted_popularity_linear, predicted_popularity_glm_ridge, predicted_popularity_glm, predicted_popularity_svm, predicted_popularity_knn)

    st.header("Predictions")

    animation(predicted_popularity, "Predicted Popularity", COLOR_GREEN)

    # panels with detail for each model
    with st.expander("See details for each model"):
        #bullet point for each model
        # keep only 2 decimals
    
        st.markdown(f"""
        - XGBoost: {round(predicted_popularity_xgb / 5, 2)}
        - Random Forest: {round(predict_popularity_rf / 5, 2)}
        - Adaboost: {round(predict_popularity_ada / 5, 2)}
        - Linear: {round(predicted_popularity_linear / 5, 2)}
        - GLM Ridge: {round(predicted_popularity_glm_ridge / 5, 2)}
        - GLM: {round(predicted_popularity_glm / 5, 2) }
        - KNN: {round(predicted_popularity_knn / 5, 2) }
        """)

    with st.expander("Feature importance - XGBoost"):
        model = pickle.load(open('models_path_regression/xgboost_regressor.sav', 'rb'))
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
                - Blue features contribute to LOW popularity (push the predicted popularity down)
                - Red features contribute to HIGH popularity.
                - f(x) is the XGBoost model's perdiction for the sample considered.''')
        # For the SHAP force plot
        shap.force_plot(explainer.expected_value / 5, shap_values / 5, sample, matplotlib=True)
        st.pyplot(bbox_inches='tight')


        # def shap_model(x):
        #     return ensemble_model(x, scaler)
        
        # explainer = shap.KernelExplainer(shap_model)
        # shap_values = explainer.shap_values(sample)
        # print(shap_values)
        # st.pyplot(shap.force_plot(explainer.expected_value, shap_values, sample, matplotlib=True), bbox_inches='tight')


            
    with st.expander("Similar songs"):

        ################################### Read in files and pickles 
        ## Read in the data for plotting 
        closest_points_all_clusters = pd.read_csv('clustering_files/For_plotting_closest_points_all_clusters_final_PCA_reverse.csv')
        df = pd.read_csv('clustering_files/spotify_data_processed_clustering.csv')
        column_buffer = df.columns

        df2 = pd.read_csv('clustering_files/data_scaled_forKmeans_columnNames.csv')
        column_buffer_kmeans = df2.columns

        ## Read in models (Kmeans, PCA)
        # Load the KMeans model
        with open('clustering_files/kmeans_model.pkl', 'rb') as file:
            kmeans_model = pickle.load(file)

        # Load the PCA model
        with open('clustering_files/pca_model.pkl', 'rb') as file:
            pca_model = pickle.load(file)

        with open('clustering_files/scaler.pkl', 'rb') as file:
            scaler_model = pickle.load(file)

        ################################### User input 
        user_input = sample

        # label from the track name
        track_name_label = 1

        # add new columns
        user_input['track_name_labels'] = track_name_label
        user_input['track_name'] = title
        user_input['popularity'] = predicted_popularity

        print(user_input)

        original_user_input = user_input # keep an original version, non standarlized for plotting 

        ################################### standarlize
        # let the user input be in the same order as data feed into standarlization
        column_order = list(column_buffer)
        user_input_ordered = user_input[column_order]

        # Normalize on non_binary data 
        user_input_ordered = user_input_ordered.drop(['track_name'], axis =1)
        binary_columns = [col for col in user_input_ordered.columns if col.startswith('genre') or col.startswith('key') or col.startswith('time_signature') or col.startswith('mod')]
        binary_data = user_input_ordered[binary_columns]
        non_binary_columns = user_input_ordered.drop(binary_columns, axis=1).columns
        non_binary_data = user_input_ordered[non_binary_columns]
        non_binary_data.head()

        # Apply the Scaler model stores to only non-binary data (excluding track name)
        non_binary_data_scaled = scaler_model.transform(non_binary_data)

        # Convert the scaled non-binary data back to a DataFrame
        non_binary_data_scaled_df = pd.DataFrame(non_binary_data_scaled, columns=non_binary_columns)

        # Reset the index for both DataFrames to ensure they align
        non_binary_data_scaled_df.reset_index(drop=True, inplace=True)
        binary_data.reset_index(drop=True, inplace=True)

        # Concatenate the DataFrames horizontally
        user_input_scaled = pd.concat([non_binary_data_scaled_df, binary_data], axis=1)

        # Display the first few rows of the combined scaled data
        user_input_scaled.head()

        ##################################### Cluster and PCA
        # reorder according to kmeans sequence
        column_order_kmeans = list(column_buffer_kmeans)
        user_input_scaled_ordered = user_input_scaled[column_order_kmeans]

        # Use the KMeans model to predict clusters for new data
        user_clusters = kmeans_model.predict(user_input_scaled_ordered)

        # Use the PCA model to transform new data to the PCA space
        user_pca = pca_model.transform(user_input_scaled_ordered)



        ################################### Reverse dummy for original_user input and add PCA and cluster label
        def reverse_one_hot(df, prefix):
            categories = [None] * df.shape[0]
            dummy_columns = [col for col in df.columns if col.startswith(prefix)]
            
            for col in dummy_columns:
                category_name = col.split(prefix + '_')[1]
                rows_with_category = df[col] == 1
                categories = [category_name if row else current for row, current in zip(rows_with_category, categories)]
            
            return pd.Series(categories, index=df.index)

        genre_column = reverse_one_hot(original_user_input, 'genre')
        original_user_input['genre'] = genre_column

        key_column = reverse_one_hot(original_user_input, 'key')
        original_user_input['key'] = key_column

        time_column = reverse_one_hot(original_user_input, 'time_signature')
        original_user_input['time_signature'] = time_column

        genre_columns_to_drop = [col for col in original_user_input.columns if 'genre_' in col]
        key_columns_to_drop = [col for col in original_user_input.columns if 'key_' in col]
        time_columns_to_drop = [col for col in original_user_input.columns if 'time_signature_' in col]
        columns_to_drop = genre_columns_to_drop + key_columns_to_drop + time_columns_to_drop
        original_user_input = original_user_input.drop(columns=columns_to_drop)

        # Print the first few rows of the updated dataframe
        original_user_input.head()

        # Add the cluster labels and  PCA components to the original_user_input DataFrame
        original_user_input['cluster_label'] = user_clusters
        original_user_input['PCA1'] = user_pca[:, 0]
        original_user_input['PCA2'] = user_pca[:, 1]

        ##################################### Plot 
        # Reorder original_user_input to match the column order of closest_points_all_clusters
        original_user_input_reordered = original_user_input[closest_points_all_clusters.columns]

        ####################
        # Assuming original_user_input_reordered and closest_points_all_clusters are existing DataFrames

        # Combine the DataFrames with an identifier
        original_user_input_reordered['source'] = 'original_user_input'
        closest_points_all_clusters['source'] = 'closest_points_all_clusters'
        stacked_dataframes = pd.concat([original_user_input_reordered, closest_points_all_clusters])

        # Assign colors: red for user input, white for all other points
        stacked_dataframes['color'] = stacked_dataframes['source'].apply(lambda x: 'red' if x == 'original_user_input' else 'rgba(0,0,0,0)')

        ## Hover information
        hover_data_user = [f'{col}: {original_user_input_reordered[col].values[0]}' for col in original_user_input_reordered.columns if col not in ['PCA1', 'PCA2']]
        hover_data = [col for col in closest_points_all_clusters.columns if col not in ['PCA1', 'PCA2']]
        hover_data_combined = [col for col in stacked_dataframes.columns if col not in ['PCA1', 'PCA2','source','color']]

        hover_text_clusters = closest_points_all_clusters.apply(
            lambda row: '<br>'.join([f'{col}: {row[col]}' for col in hover_data]),
            axis=1
        )
        hover_text_combined = stacked_dataframes.apply(
            lambda row: '<br>'.join([f'{col}: {row[col]}' for col in hover_data_combined]),
            axis=1
        )

        # Create subplot figure
        fig = go.Figure()

        # Second Plot (Cluster Points)
        fig.add_trace(
            go.Scatter(
                x=closest_points_all_clusters['PCA1'],
                y=closest_points_all_clusters['PCA2'],
                mode='markers',
                marker=dict(
                    color=closest_points_all_clusters['cluster_label'], 
                    colorscale=px.colors.sequential.Viridis,
                    opacity=0.7  # Set the opacity level here
                ),
                name='Cluster Points',
                hoverinfo='text',
                hovertext=hover_text_clusters,
                
            )
        )

        # First Plot (All Points)
        fig.add_trace(
            go.Scatter(
                x=stacked_dataframes['PCA1'],
                y=stacked_dataframes['PCA2'],
                mode='markers', 
                marker=dict(
                    color=stacked_dataframes['color'],
                    size = 15,
                ),
                name='User Input',
                hoverinfo='text',
                hovertext=hover_text_combined
            )
        )




        ########################

        # Step 1: Create the scatter plot for all cluster points
        # fig_clusters = px.scatter(
        #     closest_points_all_clusters,
        #     x='PCA1',
        #     y='PCA2',
        #     hover_data=hover_data,
        #     color='cluster_label',
        #     color_continuous_scale=px.colors.sequential.Viridis,
        #     title='Song Cluster'
        # )

        # fig_clusters = go.Figure(data=fig_clusters.data, layout=fig_clusters.layout)

        # # Step 2: Create a separate scatter plot for the user input point
        # fig_user_input = go.Figure(data=[
        #     go.Scatter(
        #         x=[original_user_input_reordered['PCA1'].values[0]],
        #         y=[original_user_input_reordered['PCA2'].values[0]],
        #         mode='markers',
        #         hovertext='<br>'.join(hover_data_user),
        #         marker=dict(color='red', size=25),
        #         name='User Input'
        #     )
        # ])


        # Step 3: Overlay the user input plot on top of the cluster plot
        # fig_clusters.add_trace(fig_user_input.data)


        # fig_clusters.data = (fig_clusters.data[0],fig_clusters.data[1])

        # fig_clusters.update_layout(autosize=True)
        # st.plotly_chart(fig_clusters, use_container_width=True)

        # Display the combined figure
        # st.plotly_chart(fig_clusters)

        # Adjust the size of the plotly figure based on sidebar input
        # width = st.sidebar.slider("Width", 500, 1500, 1000)  # Default width 1000
        # height = st.sidebar.slider("Height", 300, 900, 600)  # Default height 600

        fig.update_layout(width=1000, height=600)

        st.plotly_chart(fig, use_container_width=True)

        # Add the user input trace on top of the scatter plot
        # fig.add_trace(user_input_trace)
        # st.plotly_chart(fig)


        # # Add the user input trace to the figure
        # fig.add_trace(user_input_trace)

        # # Show the plot
        # st.plotly_chart(fig)


        ################################### Song recommendation 
        songs_scaled = pd.read_csv('clustering_files/SongRec_points.csv')
        songs_scaled.head()

        user_input_scaled_ordered['cluster_label'] = user_clusters

        # Step 1: Filter songs in the same cluster as the user's input
        user_cluster = user_input_scaled_ordered['cluster_label'].iloc[0]
        similar_songs = songs_scaled[songs_scaled['cluster_label'] == user_cluster]

        # Step 2: Compute pairwise distances
        # Drop the 'cluster_label' column for distance calculation
        distances = pairwise_distances(similar_songs.drop(['cluster_label','track_name', 'PCA1', 'PCA2'], axis=1), 
                                    user_input_scaled_ordered.drop('cluster_label', axis=1))

        # Flatten the distance array for easier handling
        distances = distances.flatten()

        # Step 3: Find indices of the three closest songs
        # Use argsort to get indices and then slice to get top 3, excluding the first one if it's zero (the song itself)
        closest_indices = np.argsort(distances)[1:4]  # Exclude the first one in case it's the song itself

        # Step 4: Retrieve the closest songs
        closest_songs = similar_songs.iloc[closest_indices]

        #### Now reverse coding, give back genre 
        genre_column = reverse_one_hot(closest_songs, 'genre')
        closest_songs['genre'] = genre_column

        key_column = reverse_one_hot(closest_songs, 'key')
        closest_songs['key'] = key_column

        time_column = reverse_one_hot(closest_songs, 'time_signature')
        closest_songs['time_signature'] = time_column

        genre_columns_to_drop = [col for col in closest_songs.columns if 'genre_' in col]
        key_columns_to_drop = [col for col in closest_songs.columns if 'key_' in col]
        time_columns_to_drop = [col for col in closest_songs.columns if 'time_signature_' in col]
        columns_to_drop = genre_columns_to_drop + key_columns_to_drop + time_columns_to_drop
        closest_songs = closest_songs.drop(columns=columns_to_drop)

        # Print the first few rows of the updated dataframe
        closest_songs.head()

        # Final Stage 
        columns_to_display = ['track_name'] # can add more, like genre yeayea 

        st.header("We identified 3 similar songs:")

        # Define the features to display for each song, including the track name
        features_to_display = ['track_name', 'genre', 'key', 'cluster_label', 'danceability', 'loudness', 'energy']

        # Create a new DataFrame with the desired columns and rounded values for numeric features
        display_df = closest_songs[features_to_display].copy()

        # Apply rounding to numeric columns
        numeric_columns = ['danceability', 'loudness', 'energy']
        for col in numeric_columns:
            display_df[col] = display_df[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

        # Reset the index to ensure it starts at 1 and increments by 1 for each row
        display_df.reset_index(drop=True, inplace=True)
        display_df.index = display_df.index + 1

        # Display the DataFrame in Streamlit
        st.dataframe(display_df, use_container_width=True)


