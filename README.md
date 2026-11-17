
# CineCompass — Movie Recommendation System

A movie discovery application built using Python, Streamlit, TMDB and NLP.

## Features

- Content-based filtering using TF-IDF and cosine similarity
- Plot sentiment matching using TextBlob
- Natural-language search
- Filters for release year, rating, vote count, genre, runtime, certification, language, actors and directors
- Similar movie recommendations
- User watchlist with personalized recommendations
- Weekly trending movies
- Movie rating and runtime visualization

## Recommendation Algorithms

The system combines:

1. Content-based similarity: 58%
2. Sentiment-based matching: 22%
3. Rating and vote-confidence score: 20%

## Technology

- Python
- Streamlit
- TMDB API
- scikit-learn
- TextBlob
- Pandas

## Deployment

The application is deployed using Streamlit Community Cloud.

Add your TMDB API Read Access Token to Streamlit secrets using the name TMDB_READ_TOKEN.

## Limitations

Recommendations are calculated from a limited group of retrieved movies. Sentiment in plot descriptions is only an approximation of movie mood.

The watchlist is saved only for the active browser session.

## Data Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
