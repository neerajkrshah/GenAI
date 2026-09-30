import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords
import os

# ---------- NLTK Setup ----------
nltk_dir = "./nltk_data"
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.append(nltk_dir)

try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', download_dir=nltk_dir)
    stop_words = set(stopwords.words('english'))

# ---------- Load Data ----------
@st.cache_resource
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv", usecols=['title', 'overview'])
    df = df.dropna(subset=['overview']).reset_index(drop=True)
    
    # Limit to 1500 movies for free Render plan
    df = df.head(1500).copy()

    def clean(text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return " ".join([w for w in text.split() if w not in stop_words and len(w) > 2])

    df['clean'] = df['overview'].apply(clean)

    tfidf = TfidfVectorizer(max_features=2500, stop_words='english')
    matrix = tfidf.fit_transform(df['clean'])
    
    # Use linear_kernel (faster than cosine_similarity)
    sim = linear_kernel(matrix, matrix)
    
    return df, sim

df, sim = load_data()

def get_recommendations(title, top_n=5):
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df['title'].iloc[i[0]] for i in scores]

# ---------- UI ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Select a Movie", sorted(df['title'].tolist()))
n = st.slider("Number of Recommendations", 3, 8, 5)

if st.button("Recommend"):
    recs = get_recommendations(movie, n)
    if recs:
        st.success(f"Recommendations for **{movie}**:")
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")
    else:
        st.warning("No recommendations found.")
        ###