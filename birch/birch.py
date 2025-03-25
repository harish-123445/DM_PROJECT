import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load

# Load the dataset
with open("birch/birch_data.pickle", "rb") as f:
    data = pickle.load(f)

# Load the trained BIRCH model
model = load('birch/birch_model.joblib')

# Standardize the features
scaler = StandardScaler()
movie_features = data.drop(columns=['title'])
movie_features_scaled = scaler.fit_transform(movie_features)

# Assign cluster labels to the dataset
data['cluster'] = model.labels_

# Function to test the BIRCH model
def birch_method_movies(movie_name, data, model, scaler, top_n=3):
    movie_name = movie_name.lower()
    matching_movies = data[data['title'].str.lower() == movie_name]

    if matching_movies.empty:
        print("No such movie found")
        return []

    movie_index = matching_movies.index[0]
    movie_features = data.drop(columns=['title', 'cluster']).iloc[movie_index].values.reshape(1, -1)
    movie_features_scaled = scaler.transform(movie_features)
    predicted_cluster = model.labels_[movie_index]

    # Filter movies from the same cluster
    cluster_movies = data[data['cluster'] == predicted_cluster]

    # Compute similarity
    similarities = cosine_similarity(movie_features_scaled, scaler.transform(cluster_movies.drop(columns=['title', 'cluster'])))
    similar_movie_indices = similarities.argsort()[0][-top_n-1:-1][::-1]

    similar_movies = cluster_movies.iloc[similar_movie_indices]['title'].tolist()
    return similar_movies

def find_similar_movies(movie_name):
    similar_movies = birch_method_movies(movie_name, data, model, scaler, top_n=5)
    return similar_movies
