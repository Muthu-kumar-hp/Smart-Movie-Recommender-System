import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

# Global variables
movies_df = None
tfidf_matrix = None
cosine_sim = None
vectorizer = None
genre_model = None

def safe_literal_eval(val):
    if pd.isna(val) or val == '':
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def extract_names(data, key='name', limit=3):
    if isinstance(data, list):
        return [item.get(key, '') for item in data[:limit] if isinstance(item, dict)]
    return []

def preprocess_data(csv_path='tmdb_5000_movies.csv'):
    global movies_df, tfidf_matrix, cosine_sim, vectorizer, genre_model

    try:
        movies_df = pd.read_csv(csv_path)
        movies_df = movies_df.fillna('')

        for col in ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']:
            if col in movies_df.columns:
                movies_df[col] = movies_df[col].apply(safe_literal_eval)

        movies_df['genre_names'] = movies_df['genres'].apply(lambda x: extract_names(x, 'name', 5))
        movies_df['keyword_names'] = movies_df['keywords'].apply(lambda x: extract_names(x, 'name', 10))
        movies_df['company_names'] = movies_df['production_companies'].apply(lambda x: extract_names(x, 'name', 3))

        movies_df['combined_features'] = (
            movies_df['genre_names'].apply(lambda x: ' '.join(x)) + ' ' +
            movies_df['keyword_names'].apply(lambda x: ' '.join(x)) + ' ' +
            movies_df['overview'].fillna('') + ' ' +
            movies_df['company_names'].apply(lambda x: ' '.join(x))
        )

        if not os.path.exists('vectorizer.pkl'):
            print("Error: vectorizer.pkl not found. Please run train.py first.")
            return False

        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        tfidf_matrix = vectorizer.transform(movies_df['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        if os.path.exists('genre_model.pkl'):
            with open('genre_model.pkl', 'rb') as f:
                genre_model = pickle.load(f)
            print("Genre model loaded successfully.")
        else:
            print("Warning: genre_model.pkl not found. Genre prediction will not be available.")

        print(f"Data loaded successfully. Total movies: {len(movies_df)}")
        return True

    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def get_recommendations(title, num_recommendations=10):
    global movies_df, cosine_sim

    try:
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index
        if len(idx) == 0:
            partial_matches = movies_df[movies_df['title'].str.lower().str.contains(title.lower(), na=False)]
            if len(partial_matches) == 0:
                return None, "Movie not found in database"
            idx = partial_matches.index[0]
        else:
            idx = idx[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

        recommendations = []
        for i in movie_indices:
            movie = movies_df.iloc[i]
            recommendations.append({
                'title': movie['title'],
                'overview': movie['overview'][:200] + '...' if len(movie['overview']) > 200 else movie['overview'],
                'genres': ', '.join(movie['genre_names']) if movie['genre_names'] else 'N/A',
                'release_date': movie.get('release_date', 'N/A'),
                'vote_average': movie.get('vote_average', 'N/A'),
                'popularity': round(movie.get('popularity', 0), 1),
                'similarity_score': round(sim_scores[movie_indices.index(i)+1][1], 3)
            })

        return recommendations, None

    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"

def predict_genre(text):
    global vectorizer, genre_model

    if not vectorizer or not genre_model:
        return None

    try:
        text_vector = vectorizer.transform([text])
        predicted_genre = genre_model.predict(text_vector)[0]
        proba = genre_model.predict_proba(text_vector)[0]
        confidence = max(proba)

        return {
            'genre': predicted_genre,
            'confidence': round(confidence, 3)
        }
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return None
