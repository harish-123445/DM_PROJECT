import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
import pickle
import warnings
warnings.filterwarnings('ignore')
# Custom K-means implementation
class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        """Initialize centroids using k-means++ method"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Choose the first centroid randomly
        centroids = np.zeros((self.n_clusters, n_features))
        first_centroid_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_centroid_idx]

        # Choose the remaining centroids
        for k in range(1, self.n_clusters):
            # Calculate distances from points to the centroids
            distances = np.sqrt(((X - centroids[:k, np.newaxis])**2).sum(axis=2))
            # Distances to closest centroid
            closest_dist = distances.min(axis=0)
            # Probability proportional to squared distance
            weights = closest_dist**2
            # Choose next centroid
            next_centroid_idx = np.random.choice(n_samples, p=weights/weights.sum())
            centroids[k] = X[next_centroid_idx]

        return centroids

    def _get_closest_centroid(self, X, centroids):
        """Assign each sample to the closest centroid"""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _get_centroids(self, X, labels):
        """Update centroids based on mean of points in each cluster"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:  # Avoid empty clusters
                centroids[k] = np.mean(X[labels == k], axis=0)
        return centroids

    def _compute_inertia(self, X, labels, centroids):
        """Calculate sum of squared distances of samples to their closest centroid"""
        distances = np.sqrt(((X - centroids[labels])**2).sum(axis=1))
        return np.sum(distances**2)

    def fit(self, X):
        """Fit K-means to the data"""
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)

        # Main K-means loop
        for i in range(self.max_iter):
            # Assign samples to closest centroids
            old_labels = self.labels_ if self.labels_ is not None else None
            self.labels_ = self._get_closest_centroid(X, self.centroids)

            # Update centroids
            self.centroids = self._get_centroids(X, self.labels_)

            # Check for convergence
            if old_labels is not None and np.all(old_labels == self.labels_):
                break

        # Calculate final inertia
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)
        return self

    def predict(self, X):
        """Predict the closest cluster for each sample in X"""
        return self._get_closest_centroid(X, self.centroids)

# Main process for movie recommendation system
def create_movie_recommendation_system(data, meta=None):
    # Extracting the movie title and features
    movie_titles = data['title']
    feature_columns = [col for col in data.columns if col != 'title']
    movie_features = data[feature_columns]

    # Scaling the features
    scaler = StandardScaler()
    movie_features_scaled = scaler.fit_transform(movie_features)

    # Determining the best k-values using elbow method
    wcss = []
    k_range = range(1, 100)
    for i in k_range:
        kmeans = CustomKMeans(n_clusters=i, random_state=42)
        kmeans.fit(movie_features_scaled)
        wcss.append(kmeans.inertia_)

    # Plotting the Elbow method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(0, 101, 5))
    plt.xlim(0, 100)
    plt.grid(True)
    plt.savefig('elbow_method.png')
    plt.close()

    # Using k=34 as mentioned in the code
    k = 34
    kmeans = CustomKMeans(n_clusters=k, random_state=42)
    kmeans.fit(movie_features_scaled)

    # Adding cluster labels to the original dataset
    data['cluster'] = kmeans.labels_

    # Saving the model and necessary components
    model_components = {
        'kmeans': kmeans,
        'scaler': scaler,
        'data': data,
        'movie_features_scaled': movie_features_scaled,
        'feature_columns': feature_columns  # Save the feature columns used
    }

    with open('movie_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model_components, f)

    # If meta data is provided, calculate evaluation metrics
    if meta is not None:
        evaluate_clustering(movie_features_scaled, kmeans.labels_, meta)

    # Visualize clusters using PCA
    visualize_clusters(movie_features_scaled, kmeans.labels_)

    return model_components

def evaluate_clustering(features, labels, meta):
    # Getting True Labels
    true_labels = meta['title'].tolist()

    # Calculating the silhouette score
    silhouette_avg = silhouette_score(features, labels)
    print("Silhouette Score:", silhouette_avg)

    # Calculating the Calinski-Harabasz index
    ch_score = calinski_harabasz_score(features, labels)
    print("Calinski-Harabasz Index:", ch_score)

    # Calculating the Davies-Bouldin index
    db_score = davies_bouldin_score(features, labels)
    print("Davies-Bouldin Index:", db_score)

    # Calculating homogeneity, completeness, and V-measure
    homogeneity = homogeneity_score(true_labels, labels)
    completeness = completeness_score(true_labels, labels)
    v_measure = v_measure_score(true_labels, labels)

    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)

def visualize_clusters(features, labels):
    # Reducing the feature dimensions to 2 for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Plotting the clusters
    plt.figure(figsize=(10, 8))

    # Scatter plot for each cluster
    for cluster_label in set(labels):
        cluster_mask = (labels == cluster_label)
        plt.scatter(features_2d[cluster_mask, 0], features_2d[cluster_mask, 1],
                   label=f'Cluster {cluster_label}', alpha=0.7)

    plt.title('Clusters of Movies (Custom K-means)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              borderaxespad=0., title='Clusters', ncol=2)
    plt.savefig('cluster_visualization.png')
    plt.close()

def recommend_similar_movies(movie_name, model_components, top_n=3):
    # Unpack model components
    kmeans = model_components['kmeans']
    scaler = model_components['scaler']
    data = model_components['data']
    movie_features_scaled = model_components['movie_features_scaled']

    # Find the movie index
    try:
        mv_index = data[data['title'] == movie_name].index.values[0]
    except IndexError:
        print(f"Movie '{movie_name}' not found in the dataset.")
        return []

    # Instead of dynamically determining feature columns here,
    # we should use the exact same features that were used during training

    # SOLUTION 1: If movie_features_scaled was created with a specific set of columns:
    # Get the original columns used for scaling during model creation
    if 'feature_columns' in model_components:
        feature_columns = model_components['feature_columns']
    else:
        # As a fallback, use what's available but print a warning
        feature_columns = [col for col in data.columns if col not in ['title', 'cluster']]
        print("Warning: Using dynamically determined features which may not match training features")

    # Get movie features
    mv_features = data.loc[mv_index, feature_columns].values.reshape(1, -1)

    # Debug: Check feature dimensions
    print(f"Model expects {scaler.n_features_in_} features, providing {mv_features.shape[1]}")

    # Make sure we have the same number of features as during training
    if mv_features.shape[1] != scaler.n_features_in_:
        print(f"Feature mismatch: Model expects {scaler.n_features_in_} features but got {mv_features.shape[1]}")
        print("Missing features may include: ")
        # This would help identify which feature is missing
        # But would require knowing the original training features
        return []

    mv_scaled = scaler.transform(mv_features)

    # Predict the cluster
    predicted_cluster = kmeans.predict(mv_scaled)[0]

    # Get movies in the same cluster
    cluster_movies = data[data['cluster'] == predicted_cluster]

    # Calculate similarities, but maintain the same feature set
    cluster_features = cluster_movies[feature_columns].values
    cluster_features_scaled = scaler.transform(cluster_features)
    similarities = cosine_similarity(mv_scaled, cluster_features_scaled)[0]

    # Sort by similarity and get top_n (exclude the movie itself)
    similar_indices = np.argsort(similarities)[-top_n-1:-1][::-1]

    # Handle case where there might not be enough movies in the cluster
    if len(similar_indices) < top_n:
        print(f"Warning: Only found {len(similar_indices)} similar movies in cluster {predicted_cluster}")

    similar_movies = cluster_movies.iloc[similar_indices]['title'].tolist()

    return similar_movies

def find_similar_movies(movie_name):
    with open('k_means/k_means.pkl', 'rb') as f:
        model_components = pickle.load(f)
    # Get recommendations
    similar_movies = recommend_similar_movies(movie_name, model_components)

    return similar_movies
