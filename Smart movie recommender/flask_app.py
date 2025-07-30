from flask import Flask, render_template, request, jsonify
import data_utils
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify([])

    matches = data_utils.movies_df[data_utils.movies_df['title'].str.lower().str.contains(query.lower(), na=False)]
    results = matches['title'].head(10).tolist()
    return jsonify(results)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        return render_template('recommend.html')

    movie_title = request.form.get('movie_title', '').strip()
    if not movie_title:
        return render_template('recommend.html', error="Please enter a movie title")

    recommendations, error = data_utils.get_recommendations(movie_title)
    if error:
        return render_template('recommend.html', error=error, movie_title=movie_title)

    return render_template('recommend.html',
                           recommendations=recommendations,
                           movie_title=movie_title)

@app.route('/api/recommend/<movie_title>')
def api_recommend(movie_title):
    recommendations, error = data_utils.get_recommendations(movie_title)
    if error:
        return jsonify({'error': error}), 404

    return jsonify({
        'movie': movie_title,
        'recommendations': recommendations
    })

@app.route('/api/predict-genre', methods=['POST'])
def api_predict_genre():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = data_utils.predict_genre(text)
    if not prediction:
        return jsonify({'error': 'Genre prediction not available'}), 500

    return jsonify(prediction)

if __name__ == '__main__':
    print("Starting Movie Recommendation System...")
    print("Loading movie data and models...")

    if data_utils.preprocess_data():
        if data_utils.movies_df is not None:
            print(f"✓ Data loaded successfully! Total movies: {len(data_utils.movies_df)}")
        else:
            print("✓ Data loaded, but movies_df is None.")
        print("✓ Starting Flask app on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("✗ Failed to load data. Please ensure you have:")
        print("  - tmdb_5000_movies.csv")
        print("  - vectorizer.pkl")
        print("  - genre_model.pkl")
