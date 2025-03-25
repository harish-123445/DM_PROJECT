# app.py
from flask import Flask, render_template, send_from_directory, request, jsonify

import csv
from flask import Flask, render_template, request
import csv
from k_means.k_means import CustomKMeans
from find_movies import find_similar_movie 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/eda')
def eda():
    images = ['Assets/1.png', 'Assets/2.png', 'Assets/3.png', 'Assets/4.png', 'Assets/5.png', 'Assets/6.png', 'Assets/7.png', 'Assets/8.png', 'Assets/9.png', 'Assets/10.png']
    return render_template('EDA.html', images=images)

@app.route('/movies.csv')
def serve_csv():
    return send_from_directory(app.root_path, 'movies.csv')

@app.route('/results', methods=('POST', 'GET'))
def result():
    if request.method == 'GET':
        movie = request.args.get('movie')
        pred_mov = find_similar_movie(movie)
        if isinstance(pred_mov, str):
            return render_template('result.html', error=pred_mov, org = movie)
        else:
            movie_details = []
            pred_mov = [title.lower() for title in pred_mov]
            with open('movies.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['title'].lower() in pred_mov:
                        movie_details.append({
                            'title': row['title'],
                            'type': row['type'],
                            'date_added': row['date_added'],
                            'release_year': row['release_year'],
                            'rating': row['rating'],
                            'description': row['description']
                        })
            return render_template('result.html', movies=movie_details, org = movie.upper())

@app.route('/image/<int:image_index>')
def full_screen(image_index):
    image_url = 'Assets/{}.png'.format(image_index)
    return render_template('full_screen.html', image=image_url)

if __name__ == '__main__':
    app.run(debug=True)