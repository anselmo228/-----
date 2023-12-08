import joblib
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
#from sagemaker_inference import content_types
from sagemaker.predictor import Predictor
import json
import boto3

# Load game data
data = pd.read_csv("./final_ITEM_DATA1.csv")

# Load user data
userData = pd.read_csv("./final_user_data.csv")

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

def calculate_weighted_average(user_games_info, data, features):
    weighted_averages = {}
    
    for feature in features:
        feature_weighted_sum = 0

        for idx, game in user_games_info.iterrows():
            game_data = data[data['Name'] == game['gameName']][feature]
            
            # Process only if game data exists
            if not game_data.empty:
                game_weight = game['percentage_playTime']
                feature_weighted_sum += (game_data.values[0] * game_weight)
                

        weighted_averages[feature] = feature_weighted_sum

    return pd.DataFrame(weighted_averages, index=[0])

class ClusteringPredictor(Predictor):
    def __init__(self, model, scaler, pca, features):
        self.scaler = scaler
        self.pca = pca
        self.features = features
        self.model = model

    def predict(self, user_data, content_type):
        
        user_data = json.loads(user_data)
        user_id = user_data.get('userId', None)
        if user_id is None:
            raise ValueError("Invalid input. 'userId' is missing.")

        # Extract and print user_id only when making predictions
        if content_type == 'application/json':
            print(f"User ID: {user_id}")
        
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

    

model_file_path = './clustering.pkl'
kmeans_model = joblib.load(model_file_path)


# 모델과 추론 스크립트 연결
model = kmeans_model # 학습된 클러스터링 모델의 경로
my_predictor = ClusteringPredictor(model=model, scaler=scaler, pca=pca, features=features, instance_type='ml.t3.medium')

# 추론 요청
userId_data = {'userId': 53875128}  # 사용자 아이디 등 추론에 필요한 데이터
response = my_predictor.predict(json.dumps(userId_data), content_type='application/json')
print(response)