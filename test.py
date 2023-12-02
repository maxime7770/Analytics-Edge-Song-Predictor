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
        width = st.sidebar.slider("Width", 500, 1500, 1000)  # Default width 1000
        height = st.sidebar.slider("Height", 300, 900, 600)  # Default height 600

        fig.update_layout(width=width, height=height)

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

        st.header("Here are three songs for you, Enjoy!:")

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


