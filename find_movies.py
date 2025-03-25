from agglomerative_model.agglomerative_model import find_similar_movie as find_similar_movie_agglomerative
from dbscan.dbscan import find_similar_movie as find_similar_movie_dbscan
from gmm_model.gmm_model import find_similar_movies as find_similar_movie_gmm
from mean_shift.mean_shift import find_similar_movies as find_similar_movie_mean_shift
from birch.birch import find_similar_movies as find_similar_movie_birch
from k_means.k_means import CustomKMeans
from k_means.k_means import find_similar_movies as find_similar_movie_kmeans



def find_similar_movie(movie_name):
    final_lst=set()
    final_lst.update(find_similar_movie_agglomerative(movie_name))
    final_lst.update(find_similar_movie_dbscan(movie_name))
    final_lst.update(find_similar_movie_gmm(movie_name))
    final_lst.update(find_similar_movie_mean_shift(movie_name))
    final_lst.update(find_similar_movie_birch(movie_name))
    final_lst.update(find_similar_movie_kmeans(movie_name))
    print(final_lst)
    return list(final_lst)

