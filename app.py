# ============================================
# Optimized Movie Recommendation System
# Faster loading for Render
# ============================================

import streamlit as st
import pandas as pd
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    stop_words = set(nltk.corpus.stopwords.words('english'))

@st.cache_resource
def load_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    
    # Keep only necessary columns and drop missing overviews
    df = df[['title', 'overview']].dropna(subset=['overview']).reset_index(drop=True)
    
    # Take only first 2000 movies to make it faster (optional but recommended)
    df = df.head(2000).copy()

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
        return " ".join(tokens)

    df['clean_text'] = df['overview'].apply(clean_text)

    # TF-IDF with fewer features for speed
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 1), min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    cosine_sim = cosine_similarity(tfidf_matrix)

    return df, cosine_sim

df, cosine_sim = load_data()

def recommend(movie_name, top_n=5):
    matches = df[df['title'].str.lower() == movie_name.lower()]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in scores]
    return df['title'].iloc[indices].tolist()

# ------------------ UI ------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommendations using movie overviews")

movie_list = sorted(df['title'].tolist())
selected = st.selectbox("Select a Movie", movie_list)

top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Get Recommendations"):
    recs = recommend(selected, top_n)
    
    if recs:
        st.success(f"Movies similar to **{selected}**:")
        for i, m in enumerate(recs, 1):
            st.write(f"**{i}. {m}**")
    else:
        st.warning("No recommendations found.")
        ########################