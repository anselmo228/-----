import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean
from sagemaker.predictor import Predictor
import json
import boto3


data = st.file_uploader('Upload Game Data CSV', type=['csv'])
game_data = load_data(data_file)

# Load user data
user_data = st.file_uploader('Upload User Data CSV', type=['csv'])
user_data = load_data(user_data_file)

# Get unique gameName values from another CSV file
valid_game_names = userData['gameName'].unique()

# Filter rows in 'data' dataframe to keep only those with gameName in valid_game_names
data = data[data['Name'].isin(valid_game_names)]

# Select features like genre and positive, negative, price
features = [
    'Positive', 'Negative', 'Recommendations', 'Peak CCU', 'Estimated owners', 'Price',
    'Action', 'Adventure', 'Animation & Modeling', 'Audio Production', 'Casual',
    'Design & Illustration', 'Documentary', 'Early Access', 'Education', 'Episodic', 'Free to Play',
    'Game Development', 'Gore', 'Indie', 'Massively Multiplayer', 'Movie', 'Nudity', 'Photo Editing', 'RPG',
    'Racing', 'Sexual Content', 'Short', 'Simulation', 'Software Training', 'Sports', 'Strategy',
    'Tutorial', 'Utilities', 'Video Production', 'Violent', 'Web Publishing'
]

# Select necessary columns
selected_features = data[features]

# Data normalization
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(selected_features)

# Reduce data to 2 dimensions using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Load the model
bucket_name = 'steambckts'
s3_client = boto3.client('s3')
s3_client.download_file(bucket_name, 'models/clustering.pkl', 'clustering.pkl')
kmeans_model = joblib.load('clustering.pkl')

# Define ClusteringPredictor class
class ClusteringPredictor(Predictor):
    def __init__(self, model, scaler, pca, features, **kwargs):
        super().__init__(model, **kwargs)
        self.scaler = scaler
        self.pca = pca
        self.features = features

    def predict(self, user_data, content_type):
        user_data = json.loads(user_data)
        user_id = user_data.get('userId', None)
        if user_id is None:
            raise ValueError("Invalid input. 'userId' is missing.")

        # Get information on all games played by the selected user (only game name and play time)
        user_games_info = userData[userData['userId'] == user_id][['gameName', 'playTime']]

        # List of names of all games played by the selected user
        user_games = user_games_info['gameName'].tolist()

        # Get information on all games
        user_game_info = data[data['Name'].isin(user_games)]

        # Scale playtime to a percentage between 0 and 1
        sum_play_time = user_games_info['playTime'].sum()
        user_games_info['percentage_playTime'] = user_games_info['playTime'] / sum_play_time

        # Calculate weighted average for each feature
        user_game_average = calculate_weighted_average(user_games_info, data, features)

        # set test data to average game data of user
        test_data = user_game_average.copy()

        # Data normalization
        scaled_test_data = self.scaler.transform(test_data)
        pca_test_result = self.pca.transform(scaled_test_data)

        # Predict the cluster of new data
        test_cluster = kmeans_model.predict(pca_test_result)[0]

        # Extract data points from the selected cluster
        cluster_indices = np.where(kmeans_model.labels_ == test_cluster)[0]
        cluster_data_points = pca_result[cluster_indices]

        # Calculate Spearman correlation coefficients and distances for each data point in the cluster
        spearman_distances = []
        for idx, point in enumerate(cluster_data_points):
            if np.array_equal(point, pca_test_result[0]):
                continue  # Skip the test data point itself

            # Calculate Spearman correlation coefficient
            spearman_coeff, _ = spearmanr(selected_features.iloc[cluster_indices[idx]], test_data.values[0])

            # Calculate Euclidean distance between points
            euclidean_dist = euclidean(point, pca_test_result[0])

            # Append tuple containing index, Spearman coefficient, and Euclidean distance
            spearman_distances.append((cluster_indices[idx], spearman_coeff, euclidean_dist))

        # Sort by Spearman coefficient in descending order
        spearman_distances.sort(key=lambda x: x[1], reverse=True)

        # Extracting game name and ranking information from top_similar_games based on Spearman correlation coefficient
        top_similar_games = []
        for i, (index, spearman_coeff, euclidean_dist) in enumerate(spearman_distances[:20], 1):
            game_name = data.iloc[index]['Name']
            top_similar_games.append({"Game": game_name, "Ranking": i})

        # Sort by Euclidean distance in ascending order
        spearman_distances.sort(key=lambda x: x[2])

        # Extracting game name and ranking information from top_similar_games_euclidean based on Euclidean distance
        top_similar_games_euclidean = []
        for i, (index, spearman_coeff, euclidean_dist) in enumerate(spearman_distances[:20], 1):
            game_name = data.iloc[index]['Name']
            top_similar_games_euclidean.append({"Game": game_name, "Ranking": i})

        return top_similar_games_euclidean, 'application/json'

# Define Streamlit app
def main():
    st.title('Game Recommendation App')
    user_id = st.text_input('Enter User ID:')
    if st.button('Get Recommendations'):
        user_data = {'userId': int(user_id)}
        response = my_predictor.predict(json.dumps(user_data), content_type='application/json')
        st.write(response)

# Run Streamlit app
if __name__ == '__main__':
    main()
