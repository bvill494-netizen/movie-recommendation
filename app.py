
"""CineCompass: TMDB movie search plus explainable hybrid recommendations."""
import os
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from query_parser import parse_query
from recommender import cast_names, director_names, genre_names, polarity, rank_movies
from tmdb_client import (
    TMDBError, details, discover, enrich, genres, poster_url, search_movies,
    search_person, similar_movie_candidates, trending, us_certification,
)

st.set_page_config(page_title="CineCompass", page_icon="🎬", layout="wide")
st.title("🎬 CineCompass")
st.caption("Find your next movie using story similarity, plot sentiment, and TMDB data.")
st.caption("Data from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")
if Path("assets/tmdb_logo.png").exists():
    st.image("assets/tmdb_logo.png", width=150)


def get_secret(name):
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


TOKEN = get_secret("TMDB_READ_TOKEN")
KEY = get_secret("TMDB_API_KEY")
if not (TOKEN or KEY):
    st.error("Add TMDB_READ_TOKEN to Streamlit Cloud → App settings → Secrets.")
    st.stop()

try:
    genre_options = {g["name"]: g["id"] for g in genres(TOKEN, KEY)}
except TMDBError as error:
    st.error(str(error))
    st.stop()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}
for name in ("recommendations", "similar", "watchlist_matches", "trending"):
    if name not in st.session_state:
        st.session_state[name] = []
if "seed" not in st.session_state:
    st.session_state.seed = ""


def show_movies(movies, prefix):
    if not movies:
        st.info("No movies found. Try less restrictive filters or a different title.")
        return
    st.caption("Recommendations rank a small TMDB candidate pool, not the entire movie catalog.")
    for start in range(0, min(len(movies), 12), 3):
        cols = st.columns(3)
        for column, movie in zip(cols, movies[start:start + 3]):
            with column:
                with st.container(border=True):
                    poster = poster_url(movie)
                    if poster:
                        st.image(poster, use_container_width=True)
                    st.markdown(f"**{movie.get('title', 'Untitled')} ({(movie.get('release_date') or '?')[:4]})**")
                    st.caption(
                        f"⭐ {movie.get('vote_average', 0):.1f}/10 | "
                        f"{movie.get('vote_count', 0):,} votes | "
                        f"{movie.get('runtime') or '?'} min | {us_certification(movie)} (US)"
                    )
                    st.write(", ".join(genre_names(movie)) or "Genre unavailable")
                    with st.expander("Details and why it matches"):
                        st.write(movie.get("overview") or "No plot listed")
                        st.write("Director:", ", ".join(director_names(movie)) or "Unknown")
                        st.write("Cast:", ", ".join(cast_names(movie)) or "Unknown")
                        revenue = movie.get("revenue", 0)
                        st.write("Box office:", f"${revenue:,.0f}" if revenue else "Unavailable")
                        if movie.get("_why"):
                            st.info(movie["_why"])
                        st.markdown(f"[Movie details on TMDB](https://www.themoviedb.org/movie/{movie['id']})")
                    mid = movie["id"]
                    if mid in st.session_state.watchlist:
                        if st.button("Remove from watchlist", key=f"remove-{prefix}-{mid}"):
                            del st.session_state.watchlist[mid]
                            st.rerun()
                    elif st.button("＋ Add to watchlist", key=f"add-{prefix}-{mid}"):
                        st.session_state.watchlist[mid] = movie
                        st.rerun()


page = st.radio("Navigation", ["Discover", "Similar movies", "Watchlist", "Trending"], horizontal=True)

if page == "Discover":
    st.subheader("Discover movies")
    with st.form("filters"):
        query = st.text_input("Describe what you want", "", placeholder="dark sci-fi after 2010 about space")
        selected_genres = st.multiselect("Genres", sorted(genre_options))
        genre_logic = st.radio("Combine genres using", ["OR", "AND"], horizontal=True)
        years = st.slider("Year or decade range", 1900, date.today().year + 5, (1980, date.today().year + 5))
        period = st.selectbox("Release period", ["Any", "Last 12 months", "Last 5 years", "Upcoming"])
        rating = st.slider("Minimum TMDB rating", 0.0, 10.0, 6.0, 0.5)
        votes = st.number_input("Minimum votes", 0, 1000000, 100, 50)
        runtime = st.selectbox("Runtime", ["Any", "Under 90 minutes", "90–120 minutes", "Over 120 minutes"])
        certification = st.selectbox("US certification", ["Any", "G", "PG", "PG-13", "R", "NC-17"])
        languages = {"Any": None, "English": "en", "Spanish": "es", "French": "fr", "Korean": "ko", "Japanese": "ja", "Hindi": "hi"}
        language = st.selectbox("Original language", list(languages))
        actor = st.text_input("Actor (optional)")
        director = st.text_input("Director (optional)")
        mood = st.select_slider("Desired plot tone", ["Dark", "Neutral", "Uplifting"], "Neutral")
        search = st.form_submit_button("Find my movies", type="primary")

    if search:
        try:
            parsed = parse_query(query)
            chosen = selected_genres or [g for g in parsed["genres"] if g in genre_options]
            low = max(years[0], parsed["year_min"] or years[0])
            high = min(years[1], parsed["year_max"] or years[1])
            if low > high:
                raise ValueError("The date ranges contradict each other.")
            start, end = f"{low}-01-01", f"{high}-12-31"
            today = date.today()
            if period == "Last 12 months":
                start, end = max(start, (today - timedelta(days=365)).isoformat()), min(end, today.isoformat())
            elif period == "Last 5 years":
                start, end = max(start, (today - timedelta(days=1826)).isoformat()), min(end, today.isoformat())
            elif period == "Upcoming":
                start = max(start, today.isoformat())
            if start > end:
                raise ValueError("The release period does not overlap with your year range.")
            params = {
                "sort_by": "popularity.desc", "include_adult": "false",
                "primary_release_date.gte": start, "primary_release_date.lte": end,
                "vote_average.gte": max(rating, parsed["rating_min"] or 0),
                "vote_count.gte": votes,
            }
            if chosen:
                joiner = "," if genre_logic == "AND" else "|"
                params["with_genres"] = joiner.join(str(genre_options[g]) for g in chosen)
            if languages[language]:
                params["with_original_language"] = languages[language]
            if certification != "Any":
                params.update(region="US", certification_country="US", certification=certification)
            if runtime == "Under 90 minutes":
                params["with_runtime.lte"] = 89
            elif runtime == "90–120 minutes":
                params.update({"with_runtime.gte": 90, "with_runtime.lte": 120})
            elif runtime == "Over 120 minutes":
                params["with_runtime.gte"] = 121
            for person, field, department in [(actor, "with_cast", "Acting"), (director, "with_crew", "Directing")]:
                if person.strip():
                    match = search_person(person, TOKEN, KEY, department)
                    if match is None:
                        raise ValueError(f"Could not find person: {person}")
                    params[field] = str(match["id"])
            with st.spinner("Retrieving movies and computing recommendations..."):
                candidates = discover(params, TOKEN, KEY, pages=2)
                movies = enrich(candidates, TOKEN, KEY, limit=28)
                mood_value = {"Dark": 0.0, "Neutral": 0.5, "Uplifting": 1.0}[mood]
                st.session_state.recommendations = rank_movies(movies, query=query, mood=mood_value)
        except (TMDBError, ValueError) as error:
            st.error(str(error))
    if st.session_state.recommendations:
        st.subheader("Your matches")
        show_movies(st.session_state.recommendations, "discover")
        df = pd.DataFrame([{"Runtime": m.get("runtime"), "Rating": m.get("vote_average"), "Votes": m.get("vote_count")}
                           for m in st.session_state.recommendations if m.get("runtime")])
        if not df.empty:
            st.subheader("Rating vs. runtime")
            st.scatter_chart(df, x="Runtime", y="Rating", size="Votes")
    elif search:
        st.info("No matching movies. Reduce the constraints and search again.")

elif page == "Similar movies":
    st.subheader("Find similar movies")
    with st.form("movie_title"):
        title = st.text_input("Enter a movie title", placeholder="Interstellar")
        find = st.form_submit_button("Find similar movies")
    if find:
        if not title.strip():
            st.warning("Enter a movie title.")
        else:
            try:
                matches = search_movies(title, TOKEN, KEY)
                if not matches:
                    raise ValueError("Movie title not found.")
                movie = details(matches[0]["id"], TOKEN, KEY)
                candidates = similar_movie_candidates(movie["id"], TOKEN, KEY)
                rich = enrich(candidates, TOKEN, KEY, limit=20)
                st.session_state.similar = rank_movies(rich, mood=(polarity(movie) + 1) / 2,
                                                       liked_movies=[movie], exclude_ids={movie["id"]})
                st.session_state.seed = movie["title"]
            except (TMDBError, ValueError) as error:
                st.error(str(error))
    if st.session_state.seed:
        st.write("Based on:", st.session_state.seed)
        show_movies(st.session_state.similar, "similar")

elif page == "Watchlist":
    st.subheader("Your watchlist (saved for this browser session)")
    saved = list(st.session_state.watchlist.values())
    if saved:
        show_movies(saved, "saved")
        if st.button("Recommend from my watchlist", type="primary"):
            try:
                ids = sorted({g["id"] for m in saved for g in m.get("genres", []) if g.get("id")})
                params = {"sort_by": "popularity.desc", "vote_count.gte": 100, "include_adult": "false"}
                if ids:
                    params["with_genres"] = "|".join(str(i) for i in ids[:6])
                candidates = enrich(discover(params, TOKEN, KEY), TOKEN, KEY, limit=28)
                avg_mood = (1 + sum(polarity(m) for m in saved) / len(saved)) / 2
                st.session_state.watchlist_matches = rank_movies(
                    candidates, liked_movies=saved, mood=avg_mood, exclude_ids=set(st.session_state.watchlist)
                )
            except TMDBError as error:
                st.error(str(error))
        if st.session_state.watchlist_matches:
            st.subheader("Recommended for your taste")
            show_movies(st.session_state.watchlist_matches, "personal")
    else:
        st.info("Add some movies from Discover, Similar movies or Trending.")

else:
    st.subheader("Trending movies this week")
    if st.button("Load trending movies", type="primary"):
        try:
            st.session_state.trending = enrich(trending(TOKEN, KEY), TOKEN, KEY, limit=18)
        except TMDBError as error:
            st.error(str(error))
    if st.session_state.trending:
        show_movies(st.session_state.trending, "trend")

st.divider()
st.caption("Powered by TMDB movie metadata · NLP methods: TF-IDF/cosine similarity and TextBlob sentiment")
