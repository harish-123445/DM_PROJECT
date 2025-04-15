from flask import Flask, render_template, send_from_directory, request, jsonify
import csv
import requests
from k_means.k_means import CustomKMeans
from find_movies import find_similar_movie

app = Flask(__name__)
OMDB_API_KEY = "2d970b41"  # Your OMDB API key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movies.csv')
def serve_csv():
    return send_from_directory(app.root_path, 'movies.csv')

def get_movie_poster(title):
    """Fetch movie poster from OMDB API"""
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={title}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("Response") == "True" and "Search" in data and len(data["Search"]) > 0:
        return data["Search"][0].get("Poster", "N/A")
    return None

@app.route('/results', methods=('POST', 'GET'))
def result():
    if request.method == 'GET':
        movie = request.args.get('movie')
        pred_mov = find_similar_movie(movie)
        if isinstance(pred_mov, str):
            return render_template('result.html', error=pred_mov, org=movie)
        else:
            movie_details = []
            pred_mov = [title.lower() for title in pred_mov]
            with open('movies.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['title'].lower() in pred_mov:
                        # Get movie poster from OMDB API
                        poster_url = get_movie_poster(row['title'])
                        
                        movie_details.append({
                            'title': row['title'],
                            'type': row['type'],
                            'date_added': row['date_added'],
                            'release_year': row['release_year'],
                            'rating': row['rating'],
                            'description': row['description'],
                            'poster': poster_url if poster_url and poster_url != "N/A" else None
                        })
            return render_template('result.html', movies=movie_details, org=movie.upper())

if __name__ == '__main__':
    app.run(debug=True)