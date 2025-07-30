# 🎬 Smart Movie Recommender System 🎥

This project is a content-based movie recommendation web application built using **Flask**, **Pandas**, **Scikit-learn**, and **TMDB movie metadata**. It recommends similar movies based on your favorite title using textual features like genres, keywords, production companies, and overviews.

---

## 🚀 Features

- 🔎 Search any movie from the TMDB 5000 dataset
- 🎯 Get 10 intelligent recommendations based on similarity
- 📖 View overview, genres, rating, and popularity of recommended movies
- 🌐 Simple, user-friendly web interface
- 📱 API endpoint for external integrations

---

## 🛠️ Technologies Used

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML/CSS (Jinja templates)
- JavaScript (for search autocomplete)

---

## 📁 Project Structure
smart-movie-recommender/
├── app.py # Flask application entry point
├── train.py # Script to train and save model (if needed)
├── data_utils.py # Data processing and recommendation functions
├── tmdb_5000_movies.csv # Movie dataset (from Kaggle)
├── genre_model.pkl # Optional: Saved ML model (if used)
├── vectorizer.pkl # Optional: Saved TF-IDF vectorizer
├── templates/
│ ├── index.html
│ └── recommend.html
├── static/
│ └── (optional CSS or JS files)
└── README.md

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-movie-recommender.git
cd smart-movie-recommender

2. Install Dependencies
pip install -r requirements.txt
flask
pandas
numpy
scikit-learn
python app.py

⚙️ API Usage
Endpoint:
bash
Copy
Edit
GET /api/recommend/<movie_title>
Example:
bash
Copy
Edit
GET http://localhost:5000/api/recommend/Inception

Recommendations

💡 Future Enhancements
Add user ratings for personalized recommendations

Include poster images using TMDB API

Deploy on Render / Heroku / Vercel

Add collaborative filtering with user-item matrix

📝 License
This project is licensed under the MIT License.

🤝 Contributions
Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request.

yaml
Copy
Edit

---

Let me know if you'd like a `requirements.txt`, deployment instructions (Render/Vercel/Heroku), or GitHub Actions workflow too.
