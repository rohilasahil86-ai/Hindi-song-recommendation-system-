import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# Load saved files
df = joblib.load("model/songs_data.joblib")
tfidf = joblib.load("model/tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load("model/tfidf_matrix.joblib")


# title
st.title("Hindi Song Recommendation System")

st.write("Select 5 songs and find 10 similar song recommendations.")


# all unique song titles
song_titles = df["song_title"].drop_duplicates().tolist()


# Select 5 input songs
song1 = st.selectbox("Select Song 1", song_titles)
song2 = st.selectbox("Select Song 2", song_titles)
song3 = st.selectbox("Select Song 3", song_titles)
song4 = st.selectbox("Select Song 4", song_titles)
song5 = st.selectbox("Select Song 5", song_titles)


# Recommendation button
if st.button("Recommend Songs"):

    # Store selected songs
    songs = [song1, song2, song3, song4, song5]

    # Convert selected songs to lowercase
    songs = [song.lower() for song in songs]

    # Get indexes of selected songs
    song_indices = [df[df["song_lower_title"] == song].index[0] 
                    for song in songs]

    # Calculate similarity between selected songs and all songs
    input_similarities = cosine_similarity(tfidf_matrix[song_indices], 
                                           tfidf_matrix)

    # Calculate average similarity score
    combined_scores = input_similarities.mean(axis=0)

    # Exclude the 5 input songs
    combined_scores[song_indices] = -1

    # Create index-score pairs
    songs_list = list(enumerate(combined_scores))

    # Sort based on similarity score
    songs_list = sorted(songs_list, reverse=True,key=lambda x: x[1])

    # Get top 10 recommendations
    top_songs = songs_list[:10]

    # Display recommendations
    st.subheader("Recommended Songs")

    for i in top_songs:
        st.write(df.iloc[i[0]]["song_title"])