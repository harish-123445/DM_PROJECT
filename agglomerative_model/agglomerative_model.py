import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and clustered dataset
with open('agglomerative_model/agglomerative_model.pkl', 'rb') as f:
    loaded_agglomerative = pickle.load(f)

with open('agglomerative_model/agglomerative_model_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

loaded_data = pd.read_pickle('agglomerative_model/agglomerative_model_clustered_movies.pkl')

# Function to recommend similar movies
def recommend_similar_movies(movie_name, data, agglomerative_model, scaler, top_n=3):
    if movie_name not in data['title'].values:
        return ["Movie not found in dataset."]

    movie_index = data[data['title'] == movie_name].index.values[0]
    movie_features = data.drop(columns=['title', 'cluster']).iloc[movie_index].values.reshape(1, -1)
    movie_features_scaled = scaler.transform(movie_features)
    predicted_cluster = agglomerative_model.labels_[movie_index]
    cluster_movies = data[data['cluster'] == predicted_cluster]
    similarities = cosine_similarity(movie_features_scaled, scaler.transform(cluster_movies.drop(columns=['title', 'cluster'])))
    similar_movie_indices = similarities.argsort()[0][-top_n-1:-1][::-1]
    similar_movies = cluster_movies.iloc[similar_movie_indices]['title'].tolist()
    return similar_movies

def find_similar_movie(movie_name):
    similar_movies = recommend_similar_movies(movie_name, loaded_data, loaded_agglomerative, loaded_scaler)
    return similar_movies


