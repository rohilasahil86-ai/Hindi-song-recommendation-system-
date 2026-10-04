# Hindi Song Recommendation System

A content-based Hindi song recommendation system that takes 5 input songs and recommends 10 similar songs.

## Project Overview

This project recommends songs based on song metadata such as singer, music director, lyricist, album/movie, genre, and mood.

## Dataset

- 50,000 Hindi songs
- 10 features including song title, singer, music director, lyricist, genre, mood, release year, and duration.

## Approach

1. Data preprocessing
2. Feature combination
3. TF-IDF Vectorization
4. Cosine Similarity
5. Similarity aggregation for 5 input songs
6. Top 10 song recommendations

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## How It Works

The user selects 5 songs. Their TF-IDF representations are compared with the available songs using cosine similarity. The similarity scores are averaged and the top 10 songs are recommended.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt