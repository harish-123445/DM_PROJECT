import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved Gaussian Mixture Model
with open('gmm_model/gmm_model.pkl', 'rb') as f:
    gmm_model = pickle.load(f)

# Load the saved StandardScaler
with open('gmm_model/gmm_model_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the saved clustered dataset
data = pd.read_pickle('gmm_model/gmm_model_clustered_movies.pkl')

def recommend_similar_movies(movie_name, data, gmm_model, scaler, top_n=3):
    if movie_name not in data['title'].values:
        return ["Movie not found in dataset."]

    movie_index = data[data['title'] == movie_name].index.values[0]
    movie_features = data.drop(columns=['title', 'cluster']).iloc[movie_index].values.reshape(1, -1)
    movie_features_scaled = scaler.transform(movie_features)
    predicted_cluster = gmm_model.predict(movie_features_scaled)[0]
    cluster_movies = data[data['cluster'] == predicted_cluster]
    similarities = cosine_similarity(movie_features_scaled, scaler.transform(cluster_movies.drop(columns=['title', 'cluster'])))
    similar_movie_indices = similarities.argsort()[0][-top_n-1:-1][::-1]
    similar_movies = cluster_movies.iloc[similar_movie_indices]['title'].tolist()
    return similar_movies

def find_similar_movies(movie_name):
    similar_movies = recommend_similar_movies(movie_name, data, gmm_model, scaler, top_n=5)
    return similar_movies