# ============================================
# PART 4 — Simple App Interface (Streamlit)
# Task 6: Build UI using Streamlit
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --------------------------------------------------
# Load & Preprocess Data (same as before)
# --------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df = df[['id', 'title', 'overview', 'genres']].copy()
    df['overview'] = df['overview'].fillna('')

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)

    df['clean_text'] = df['overview'].apply(preprocess_text)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    # Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim

df, cosine_sim = load_and_prepare_data()

# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------
def recommend(item_name, top_n=5):
    matches = df[df['title'].str.lower() == item_name.lower()]
    
    if matches.empty:
        return []
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices].tolist()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Content-Based Movie Recommendation System")
st.write("Select a movie and get similar movie recommendations based on plot overview.")

# Dropdown to select movie
movie_list = df['title'].sort_values().tolist()
selected_movie = st.selectbox("Select a Movie:", movie_list)

# Number of recommendations
top_n = st.slider("Number of Recommendations:", min_value=3, max_value=10, value=5)

# Button to generate recommendations
if st.button("Get Recommendations"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie, top_n)
    
    if recommendations:
        st.success(f"Top {top_n} movies similar to **{selected_movie}**:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"**{i}. {movie}**")
    else:
        st.error("Movie not found or no recommendations available.")

# Footer
st.markdown("---")
st.caption("Built with Python • Pandas • scikit-learn • Streamlit | Content-Based Filtering")