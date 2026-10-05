# Hindi Song Recommendation System

A content-based Hindi song recommendation system that takes 5 input songs and recommends 10 similar songs.

## Live Demo

https://hindi-song-recommendation-system04.streamlit.app/

## Run Locally

## Project Overview

This project recommends songs based on metadata such as singer, music director, lyricist, album/movie, genre, and mood.

## Dataset

- 50,000 Hindi songs
- 10 features including song title, singer, music director, lyricist, genre, mood, release year, and duration.

## Approach

1. Data preprocessing
2. Feature combination
3. TF-IDF Vectorization
4. Cosine Similarity
5. Similarity aggregation for 5 input songs
6. Top 10 recommendations

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## How It Works

The user selects 5 songs. Their metadata is converted into TF-IDF vectors and compared with other songs using Cosine Similarity. The similarity scores are averaged and the top 10 songs are recommended.


Install dependencies:

```bash
pip install -r requirements.txt