import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Load the saved Mean Shift model
with open('mean_shift/mean_shift_model.pkl', 'rb') as f:
    mean_shift_model = pickle.load(f)

# Load the saved StandardScaler
with open('mean_shift/mean_shift_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the saved clustered dataset
data = pd.read_pickle('mean_shift/mean_shift_clustered_movies.pkl')
def recommend_similar_movies(movie_name, data, mean_shift_model, scaler, top_n=3):
    if movie_name not in data['title'].values:
        return ["Movie not found in dataset."]

    movie_index = data[data['title'] == movie_name].index.values[0]
    
    # Extract movie features and scale them
    movie_features = data.drop(columns=['title', 'cluster']).iloc[movie_index].values.reshape(1, -1)
    movie_features_scaled = scaler.transform(movie_features)

    # Identify the cluster of the given movie
    predicted_cluster = mean_shift_model.labels_[movie_index]

    # Get movies from the same cluster
    cluster_movies = data[data['cluster'] == predicted_cluster]

    # Compute cosine similarity
    similarities = cosine_similarity(movie_features_scaled, scaler.transform(cluster_movies.drop(columns=['title', 'cluster'])))
    similar_movie_indices = similarities.argsort()[0][-top_n-1:-1][::-1]

    # Get similar movie titles
    similar_movies = cluster_movies.iloc[similar_movie_indices]['title'].tolist()
    return similar_movies

def find_similar_movies(movie_name):
    similar_movies = recommend_similar_movies(movie_name, data, mean_shift_model, scaler, top_n=5)
    return similar_movies
