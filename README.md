# Analytics Edge Project: Predicting Spotify Songs Popularity

This project was developed as part of the Advanced Analytics Edge class at MIT. It is a collaborative effort by team members [Maxime Wolf](https://www.linkedin.com/in/maxime-wolf/), [Sanya Chauhan](https://www.linkedin.com/in/sanya-chauhan/), [Vidushi Gupta](https://www.linkedin.com/in/vidushi-gupta07/), and [Xidan Xu](https://www.linkedin.com/in/xidan-xu/)

https://github.com/maxime7770/Analytics-Edge/assets/58089609/c75f0fce-b9b4-4ce9-bf39-c60735d04416

## Problem

In the contemporary music industry, aspiring artists face a challenging dilemma: predicting the potential popularity of their songs before releasing them. To address this issue, this project aims to leverage Spotify data and song attributes to assist artists in gauging the likelihood of their songs becoming popular and to compare them alongside songs with similar attributes. Also, this project revolves around song recommendation based on song features such as danceability, tempo, and musical characteristics. The problem we seek to solve is how to efficiently group songs into clusters using these intrinsic song features and recommend songs to users who prefer specific musical attributes. 

## Data Description

### Data

The "Spotify 1 Million Tracks" dataset, available on Kaggle, is meticulously curated and maintained, making it a reliable resource for data-driven music research and analysis. This dataset was extracted from Spotify, one of the world's leading music streaming platforms, using the Python library “Spotify” which allows users to access music data. It consists of popularity scores of songs, year published, and audio features, like genre, altogether 19 features for over 1 million tracks between 2000 and 2023. We retained about 5000 records for every popularity level.

### EDA

The exploratory data analysis (EDA) provides a comprehensive examination of the Spotify music tracks dataset, with a focus on genre distribution, numerical attribute analysis, popularity metrics, outlier identification, and overall data quality assessment. The genre distribution analysis highlighted the dominance of genres like k-pop, black-metal, gospel, and ambient. Histograms for attributes such as 'danceability', 'energy', 'loudness', and others depict distribution patterns for the numerical variables. The popularity analysis reveals an uneven spread, prompting a closer look at scores between 1 and 50, followed by a correlation heatmap to understand inter-attribute relationships like ‘energy’ being highly correlated with ‘acousticness’ and ‘loudness’. 
The analysis handles outliers by calculating key statistical measures like percentiles and IQR to identify and analyze data points falling outside the typical range. A pivotal part of the analysis is the genre popularity evaluation, where a bar chart showcases average popularity scores across genres, with pop emerging as the most popular genre. Lastly, the analysis addresses missing values, ensuring the dataset's robustness, and provides descriptive statistics for a comprehensive overview.

### Feature selection analysis

The feature selection analysis was conducted using a multifaceted approach. Initially, the dataset was subject to Principal Component Analysis (PCA) to reduce dimensionality and identify components capturing the most variance. The PCA revealed specific features with significant loadings, indicating their influence on the data's variation.
Further analysis was carried out using Lasso Regression (L1 Regularization), a technique known for its efficacy in both feature shrinkage and selection. The Lasso model, fitted on standardized features, identified various attributes significantly influencing the target variable. The importance of each feature was determined based on the Lasso coefficients, and a sorted list of features was created based on their coefficients' absolute values.
Additionally, a Random Forest model was employed to assess feature importance. This model's intrinsic feature_importances_attribute provided a ranking of features based on their contribution to the model's predictive power. The top features identified by the Random Forest model were found to align closely with those identified by PCA, further validating their significance.


## Analytical Techniques

### Predictive Models

We explored various regression techniques to predict the popularity of unreleased songs, leveraging a range of models from traditional linear methods to more advanced machine learning algorithms. Each model brought its unique approach to handling the dataset's complexities, with some focusing on linear relationships and others delving into more intricate patterns through ensemble and non-linear methods. Models like XGBoost, Random Forest, and AdaBoost fall into the latter category, utilizing ensemble learning to enhance prediction accuracy. Conversely, simpler models like Linear Regression and Generalized Linear Models offered a more straightforward interpretation of data features. The inclusion of K-Nearest Neighbors and Support Vector Machines added diversity to the predictive capabilities, showcasing the breadth of techniques available for tackling the multifaceted task of song popularity prediction. This variety of approaches underscored the intricate nature of predicting artistic success, where numerous factors converge to determine a song's reception.
The range of the popularity values is 0 to 50 and we train the models on this data. Here, we report the MAE on a scale of 1-10.

| Model                    | MAE      | R2     |
| ------------------------ | -------- | ------ |
| XGBoost Regressor        | 0.73632  | 0.5462 |
| Random Forest            | 0.99698  | 0.2545 |
| AdaBoost                 | 1.04349  | 0.1987 |
| Linear Regression        | 0.80339  | 0.4715 |
| Generalized Linear Model | 0.91371  | 0.3837 |
| K-Nearest Neighbors      | 0.78927  | 0.4800 |
| Support Vector Machine   | 0.75059  | 0.5011 |
| Ensemble Method using NN | 0.722885 | 0.5578 |

Ensemble methods were used to average out the predictions from the models and accordingly display one predicted value for popularity. Specifically, we use a Neural Network model trained on the stacked predictions from the previous models. This new approach leads to an improvement in the MAE and the R2 scores. It is also better than standard ensembling methods, such as taking the average of the predictions.

### Clustering

In our final segment, we apply k-means clustering to our dataset. We incorporate song features and the embedding of the track name in clustering, as we believe the song title conveys additional information, such as the song's emotional tone (e.g., sadness or happiness).To ensure a fair and unbiased contribution from each feature in the clustering process, we first standardized the dataset using StandardScaler(). This step involved normalizing each feature to have zero mean and unit variance, thus aligning all features to a common scale. 
We utilized the Elbow Method to ascertain the optimal number of clusters, k. We pick k = 60 (Appendix1: clustering) because increasing the number of clusters beyond 60 yields only marginal improvements in clustering accuracy. Then proceed to perform k-means clustering on the standardized data with k set to 60. To visualize the clusters, PCA is performed to bring down the dimensionality to two, so we plot based on the two PC values as x-axis and y-axis.  
We also recommended similar songs that leverages clustering based on musical features. Upon receiving user input, the system identifies the cluster of the input song, and then calculates Euclidean distance. Then, it presents three similar songs (in terms of euclidean distance) from within that cluster.
Then we apply streamlit to integrate everything. Utilizing a Streamlit interface, this system allows users to input their preferred song characteristics. And then visualize their input on the plot with all the other songs' data points and features as hover information. And then output three similar songs with key features. 

## Final Product and Impact

The culmination of our project was presented in an interactive Streamlit application, designed to offer both ease of use and insightful analytics for predicting song popularity. Users interact with the app by adjusting sliders to set numerical variables within their ranges and entering text-specific variables. Upon clicking the 'Predict' button, the app displays an estimated popularity score, bringing to life the complex algorithms in a user-friendly format.
A unique feature of the app is its transparency in model workings. It provides a detailed breakdown of each model's prediction process, enhancing user understanding and trust in the system. The integration of SHAP force plots is particularly noteworthy, as it visually represents the impact of different features on the predicted popularity score. These plots use color-coded indicators - blue for features decreasing popularity, and red for those increasing it - providing an intuitive understanding of each feature's influence.
Another innovative aspect is the inclusion of a clustering plot, which places the user's input in the context of existing songs. This visual representation not only locates the user's input within the larger dataset but also identifies and presents three similar songs, offering tangible comparisons and insights.
The impact of this project extends far beyond its technical achievements. For artists and record labels, this tool offers a data-driven approach to gauge potential audience reception, enabling them to make more informed decisions about song production and marketing strategies. It provides a predictive lens through which artists can anticipate market reception, potentially influencing their creative decisions. For the music industry, this represents a significant step towards harnessing the power of data analytics, blending the art of music with the science of data to better navigate the ever-evolving landscape of audience preferences and market trends.
