
"""CineCompass movie discovery website."""

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from query_parser import parse_query
from topic_search import discover_for_query

from recommender import (
    cast_names,
    director_names,
    genre_names,
    polarity,
    rank_movies,
)

from tmdb_client import (
    TMDBError,
    details,
    discover,
    enrich,
    genres,
    poster_url,
    search_movies,
    search_person,
    similar_movie_candidates,
    trending,
    us_certification,
)


# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="CineCompass | Movie Discovery",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------
# WEBSITE DESIGN - HTML FIXED
# --------------------------------

st.html("""
<style>
.cine-hero {
    padding: clamp(1.3rem, 4vw, 2.5rem);
    border: 1px solid #293446;
    border-radius: 22px;
    background:
        radial-gradient(
            circle at 91% 5%,
            rgba(248,113,113,.20),
            transparent 42%
        ),
        linear-gradient(
            120deg,
            #1c293c 0%,
            #111927 78%
        );
    margin-bottom: 1.2rem;
}

.cine-eyebrow {
    color: #ffb2a5;
    letter-spacing: .17em;
    font-weight: 800;
    font-size: .72rem;
    text-transform: uppercase;
    margin: 0 0 .5rem 0;
}

.cine-hero h1 {
    margin: 0 0 .55rem;
    font-size: clamp(2rem, 5vw, 3.3rem);
    letter-spacing: -.04em;
    line-height: 1.12;
    color: #f6f7fb;
}

.cine-subtitle {
    margin: 0;
    max-width: 45rem;
    color: #c0ccdc;
    font-size: 1.04rem;
    line-height: 1.65;
}

.cine-tags {
    margin-top: 1.25rem;
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
}

.cine-tags span {
    border: 1px solid #3a475b;
    color: #e4ebf6;
    border-radius: 50px;
    font-size: .79rem;
    padding: .28rem .7rem;
}
</style>

<div class="cine-hero">

    <p class="cine-eyebrow">
        CINECOMPASS / MOVIE DISCOVERY
    </p>

    <h1>
        Less scrolling.<br>
        More great movies.
    </h1>

    <p class="cine-subtitle">
        Describe what you want to watch.
        Discover relevant stories,
        fine-tune your results,
        and save the ones worth watching.
    </p>

    <div class="cine-tags">
        <span>✦ Smart search</span>
        <span>◉ Personal recommendations</span>
        <span>♡ Watchlist</span>
    </div>

</div>
""")


# --------------------------------
# TMDB API CONNECTION
# --------------------------------

def get_secret(name):
    try:
        return st.secrets.get(
            name,
            os.getenv(name, "")
        )
    except Exception:
        return os.getenv(name, "")


TOKEN = get_secret("TMDB_READ_TOKEN")
KEY = get_secret("TMDB_API_KEY")

if not (TOKEN or KEY):
    st.error(
        "Add your TMDB credential in "
        "Streamlit Cloud → App settings → Secrets."
    )
    st.stop()


try:
    genre_options = {
        g["name"]: g["id"]
        for g in genres(TOKEN, KEY)
    }

except TMDBError as error:
    st.error(str(error))
    st.stop()


# --------------------------------
# SESSION STATE
# --------------------------------

if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}

for name in (
    "recommendations",
    "similar",
    "watchlist_matches",
    "trending"
):
    if name not in st.session_state:
        st.session_state[name] = []

if "seed" not in st.session_state:
    st.session_state.seed = ""


# --------------------------------
# MOVIE CARDS
# --------------------------------

def show_movies(movies, prefix):

    if not movies:
        st.info(
            "No movies found. Try broadening your "
            "filters or changing your search."
        )
        return

    for start in range(
        0,
        min(len(movies), 12),
        3
    ):

        columns = st.columns(
            3,
            gap="medium"
        )

        for column, movie in zip(
            columns,
            movies[start:start + 3]
        ):

            with column:

                with st.container(border=True):

                    poster = poster_url(movie)

                    if poster:
                        st.image(
                            poster,
                            use_container_width=True
                        )
                    else:
                        st.info(
                            "🎞️ Poster unavailable"
                        )

                    year = (
                        movie.get("release_date")
                        or ""
                    )[:4] or "—"

                    rating = float(
                        movie.get("vote_average")
                        or 0
                    )

                    runtime = (
                        movie.get("runtime")
                        or "—"
                    )

                    st.markdown(
                        f"**{movie.get('title') or 'Untitled'}**"
                    )

                    st.caption(
                        f"{year} · ⭐ {rating:.1f}/10 "
                        f"· {runtime} min"
                    )

                    st.caption(
                        " / ".join(
                            genre_names(movie)[:3]
                        )
                        or "Genres not listed"
                    )

                    mid = movie["id"]

                    # Watchlist buttons

                    if mid in st.session_state.watchlist:

                        if st.button(
                            "✓ In watchlist — remove",
                            key=f"remove-{prefix}-{mid}",
                            use_container_width=True
                        ):

                            del st.session_state.watchlist[mid]
                            st.rerun()

                    else:

                        if st.button(
                            "＋ Save to watchlist",
                            key=f"add-{prefix}-{mid}",
                            use_container_width=True
                        ):

                            st.session_state.watchlist[mid] = movie
                            st.rerun()

                    # Movie details

                    with st.expander(
                        "Plot, cast & why it matches"
                    ):

                        st.write(
                            movie.get("overview")
                            or "No plot description available."
                        )

                        st.caption(
                            "Director: "
                            + (
                                ", ".join(
                                    director_names(movie)
                                )
                                or "Not listed"
                            )
                        )

                        st.caption(
                            "Cast: "
                            + (
                                ", ".join(
                                    cast_names(movie)
                                )
                                or "Not listed"
                            )
                        )

                        votes = movie.get(
                            "vote_count"
                        ) or 0

                        st.caption(
                            f"US certification: "
                            f"{us_certification(movie)} "
                            f"· {votes:,} votes"
                        )

                        revenue = (
                            movie.get("revenue")
                            or 0
                        )

                        if revenue:
                            st.caption(
                                f"Box office: ${revenue:,.0f}"
                            )
                        else:
                            st.caption(
                                "Box office: not listed"
                            )

                        if movie.get("_why"):
                            st.info(
                                movie["_why"]
                            )

                        st.markdown(
                            f"[View on TMDB]"
                            f"(https://www.themoviedb.org/movie/{mid})"
                        )


# --------------------------------
# NAVIGATION
# --------------------------------

page = st.radio(
    "Browse",
    [
        "Discover",
        "Similar movies",
        "Watchlist",
        "Trending"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# --------------------------------
# DISCOVER PAGE
# --------------------------------

if page == "Discover":

    st.subheader(
        "Find a movie for tonight"
    )

    st.caption(
        "Start with a description. Open advanced "
        "filters only if you want more control."
    )

    with st.form("filters"):

        query = st.text_input(
            "What are you in the mood for?",
            placeholder=(
                "Try: race car movies, "
                "space adventures, or a dark mystery"
            )
        )

        gcol, lcol = st.columns(
            [3, 1],
            gap="medium"
        )

        with gcol:

            selected_genres = st.multiselect(
                "Genres (optional)",
                sorted(genre_options)
            )

        with lcol:

            genre_logic = st.selectbox(
                "Genre matching",
                ["OR", "AND"]
            )

        # Advanced filters

        with st.expander(
            "⚙️ Advanced filters · year, rating, cast and more"
        ):

            a, b = st.columns(
                2,
                gap="large"
            )

            with a:

                years = st.slider(
                    "Release years",
                    1900,
                    date.today().year + 5,
                    (
                        1980,
                        date.today().year + 5
                    )
                )

                period = st.selectbox(
                    "Release period",
                    [
                        "Any",
                        "Last 12 months",
                        "Last 5 years",
                        "Upcoming"
                    ]
                )

                rating = st.slider(
                    "Minimum TMDB rating",
                    0.0,
                    10.0,
                    6.0,
                    0.5
                )

                votes = st.number_input(
                    "Minimum number of votes",
                    0,
                    1000000,
                    100,
                    50
                )

                runtime = st.selectbox(
                    "Runtime",
                    [
                        "Any",
                        "Under 90 minutes",
                        "90–120 minutes",
                        "Over 120 minutes"
                    ]
                )

            with b:

                certification = st.selectbox(
                    "US certification",
                    [
                        "Any",
                        "G",
                        "PG",
                        "PG-13",
                        "R",
                        "NC-17"
                    ]
                )

                languages = {
                    "Any": None,
                    "English": "en",
                    "Spanish": "es",
                    "French": "fr",
                    "Korean": "ko",
                    "Japanese": "ja",
                    "Hindi": "hi"
                }

                language = st.selectbox(
                    "Original language",
                    list(languages)
                )

                actor = st.text_input(
                    "Actor (optional)",
                    placeholder="e.g. Margot Robbie"
                )

                director = st.text_input(
                    "Director (optional)",
                    placeholder="e.g. Christopher Nolan"
                )

                mood = st.select_slider(
                    "Plot tone",
                    [
                        "Dark",
                        "Neutral",
                        "Uplifting"
                    ],
                    "Neutral"
                )

        search = st.form_submit_button(
            "🔎 Find my movies",
            type="primary",
            use_container_width=True
        )

    # Movie search logic

    if search:

        st.session_state.recommendations = []

        try:

            parsed = parse_query(query)

            chosen = (
                selected_genres
                or [
                    g
                    for g in parsed["genres"]
                    if g in genre_options
                ]
            )

            low = max(
                years[0],
                parsed["year_min"] or years[0]
            )

            high = min(
                years[1],
                parsed["year_max"] or years[1]
            )

            if low > high:
                raise ValueError(
                    "The date ranges contradict each other."
                )

            start = f"{low}-01-01"
            end = f"{high}-12-31"

            today = date.today()

            if period == "Last 12 months":

                start = max(
                    start,
                    (
                        today - timedelta(days=365)
                    ).isoformat()
                )

                end = min(
                    end,
                    today.isoformat()
                )

            elif period == "Last 5 years":

                start = max(
                    start,
                    (
                        today - timedelta(days=1826)
                    ).isoformat()
                )

                end = min(
                    end,
                    today.isoformat()
                )

            elif period == "Upcoming":

                start = max(
                    start,
                    today.isoformat()
                )

            if start > end:
                raise ValueError(
                    "The release period does not "
                    "overlap with your year range."
                )

            params = {
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "primary_release_date.gte": start,
                "primary_release_date.lte": end,
                "vote_average.gte": max(
                    rating,
                    parsed["rating_min"] or 0
                ),
                "vote_count.gte": votes
            }

            # Genre filters

            if chosen:

                joiner = (
                    ","
                    if genre_logic == "AND"
                    else "|"
                )

                params["with_genres"] = joiner.join(
                    str(genre_options[g])
                    for g in chosen
                )

            # Language

            if languages[language]:

                params["with_original_language"] = (
                    languages[language]
                )

            # Certification

            if certification != "Any":

                params.update(
                    region="US",
                    certification_country="US",
                    certification=certification
                )

            # Runtime

            if runtime == "Under 90 minutes":

                params["with_runtime.lte"] = 89

            elif runtime == "90–120 minutes":

                params.update({
                    "with_runtime.gte": 90,
                    "with_runtime.lte": 120
                })

            elif runtime == "Over 120 minutes":

                params["with_runtime.gte"] = 121

            # Actor and director filters

            people = [
                (
                    actor,
                    "with_cast",
                    "Acting"
                ),
                (
                    director,
                    "with_crew",
                    "Directing"
                )
            ]

            for person, field, department in people:

                if person.strip():

                    match = search_person(
                        person,
                        TOKEN,
                        KEY,
                        department
                    )

                    if match is None:
                        raise ValueError(
                            f"Could not find person: {person}"
                        )

                    params[field] = str(
                        match["id"]
                    )

            # Recommendations

            with st.spinner(
                "Finding the best movies for you..."
            ):

                candidates = discover_for_query(
                    params,
                    query,
                    TOKEN,
                    KEY
                )

                movies = enrich(
                    candidates,
                    TOKEN,
                    KEY,
                    limit=36
                )

                mood_value = {
                    "Dark": 0.0,
                    "Neutral": 0.5,
                    "Uplifting": 1.0
                }[mood]

                st.session_state.recommendations = (
                    rank_movies(
                        movies,
                        query=query,
                        mood=mood_value
                    )
                )

        except (
            TMDBError,
            ValueError
        ) as error:

            st.error(str(error))

    # Show results

    if st.session_state.recommendations:

        st.subheader(
            "Picked for you"
        )

        st.caption(
            f"{len(st.session_state.recommendations)} "
            "relevant matches · showing up to 12"
        )

        show_movies(
            st.session_state.recommendations,
            "discover"
        )

        df = pd.DataFrame([
            {
                "Runtime": m.get("runtime"),
                "Rating": m.get("vote_average"),
                "Votes": m.get("vote_count")
            }
            for m in st.session_state.recommendations
            if m.get("runtime")
        ])

        if not df.empty:

            with st.expander(
                "📊 Explore the data · rating versus runtime"
            ):

                st.scatter_chart(
                    df,
                    x="Runtime",
                    y="Rating",
                    size="Votes"
                )

    elif search:

        st.info(
            "No relevant movies found. "
            "Try lowering the rating or vote threshold, "
            "expanding the years, or rephrasing your topic."
        )


# --------------------------------
# SIMILAR MOVIES PAGE
# --------------------------------

elif page == "Similar movies":

    st.subheader(
        "More like your favourites"
    )

    st.caption(
        "Enter a movie you enjoyed and discover "
        "stories with similar themes."
    )

    with st.form("movie_title"):

        title = st.text_input(
            "Enter a movie title",
            placeholder="Interstellar"
        )

        find = st.form_submit_button(
            "Find similar movies"
        )

    if find:

        if not title.strip():

            st.warning(
                "Enter a movie title."
            )

        else:

            try:

                matches = search_movies(
                    title,
                    TOKEN,
                    KEY
                )

                if not matches:
                    raise ValueError(
                        "Movie title not found."
                    )

                movie = details(
                    matches[0]["id"],
                    TOKEN,
                    KEY
                )

                candidates = similar_movie_candidates(
                    movie["id"],
                    TOKEN,
                    KEY
                )

                rich = enrich(
                    candidates,
                    TOKEN,
                    KEY,
                    limit=20
                )

                st.session_state.similar = (
                    rank_movies(
                        rich,
                        mood=(
                            polarity(movie) + 1
                        ) / 2,
                        liked_movies=[movie],
                        exclude_ids={
                            movie["id"]
                        }
                    )
                )

                st.session_state.seed = (
                    movie["title"]
                )

            except (
                TMDBError,
                ValueError
            ) as error:

                st.error(str(error))

    if st.session_state.seed:

        st.write(
            "Based on:",
            st.session_state.seed
        )

        show_movies(
            st.session_state.similar,
            "similar"
        )


# --------------------------------
# WATCHLIST PAGE
# --------------------------------

elif page == "Watchlist":

    st.subheader(
        "Your watchlist"
    )

    st.caption(
        "Your saved picks stay available "
        "during this browser session."
    )

    saved = list(
        st.session_state.watchlist.values()
    )

    if saved:

        show_movies(
            saved,
            "saved"
        )

        if st.button(
            "Recommend from my watchlist",
            type="primary"
        ):

            try:

                ids = sorted({
                    g["id"]
                    for m in saved
                    for g in m.get("genres", [])
                    if g.get("id")
                })

                params = {
                    "sort_by": "popularity.desc",
                    "vote_count.gte": 100,
                    "include_adult": "false"
                }

                if ids:

                    params["with_genres"] = "|".join(
                        str(i)
                        for i in ids[:6]
                    )

                candidates = enrich(
                    discover(
                        params,
                        TOKEN,
                        KEY
                    ),
                    TOKEN,
                    KEY,
                    limit=28
                )

                avg_mood = (
                    1 + sum(
                        polarity(m)
                        for m in saved
                    ) / len(saved)
                ) / 2

                st.session_state.watchlist_matches = (
                    rank_movies(
                        candidates,
                        liked_movies=saved,
                        mood=avg_mood,
                        exclude_ids=set(
                            st.session_state.watchlist
                        )
                    )
                )

            except TMDBError as error:

                st.error(str(error))

        if st.session_state.watchlist_matches:

            st.subheader(
                "Inspired by your watchlist"
            )

            show_movies(
                st.session_state.watchlist_matches,
                "personal"
            )

    else:

        st.info(
            "Add some movies from Discover, "
            "Similar movies or Trending."
        )


# --------------------------------
# TRENDING PAGE
# --------------------------------

else:

    st.subheader(
        "Trending this week"
    )

    st.caption(
        "See what's capturing audiences' "
        "attention on TMDB."
    )

    if st.button(
        "Load trending movies",
        type="primary"
    ):

        try:

            st.session_state.trending = enrich(
                trending(
                    TOKEN,
                    KEY
                ),
                TOKEN,
                KEY,
                limit=18
            )

        except TMDBError as error:

            st.error(str(error))

    if st.session_state.trending:

        show_movies(
            st.session_state.trending,
            "trend"
        )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "CineCompass · Powered by TMDB data · "
    "TF-IDF content similarity + TextBlob sentiment"
)

st.caption(
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)

if Path("assets/tmdb_logo.png").exists():

    st.image(
        "assets/tmdb_logo.png",
        width=130
    )
