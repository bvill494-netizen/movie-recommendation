
"""TMDB v3 API helpers. API credentials always stay on the server."""
from __future__ import annotations

import requests
import streamlit as st

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"


class TMDBError(RuntimeError):
    """A recoverable user-facing failure while calling TMDB."""


@st.cache_data(ttl=60 * 60, show_spinner=False)
def api_get(path: str, params: dict | None = None, _token: str = "", _api_key: str = "") -> dict:
    """Cached API requests. Leading underscores exclude secrets from cache hashing."""
    if not (_token or _api_key):
        raise TMDBError("TMDB credentials have not been configured.")
    query = dict(params or {})
    headers = {"accept": "application/json"}
    if _token:
        headers["Authorization"] = f"Bearer {_token}"
    else:
        query["api_key"] = _api_key
    try:
        response = requests.get(f"{BASE_URL}/{path.lstrip('/')}", params=query, headers=headers, timeout=15)
        if response.status_code == 401:
            raise TMDBError("TMDB rejected the API credentials. Check your Streamlit secrets.")
        if response.status_code == 429:
            raise TMDBError("TMDB is temporarily rate-limiting requests. Retry shortly.")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise TMDBError(f"Couldn't retrieve data from TMDB: {exc}") from exc
    except ValueError as exc:
        raise TMDBError("TMDB returned an unreadable response.") from exc


def genres(token: str, key: str) -> list[dict]:
    return api_get("genre/movie/list", _token=token, _api_key=key).get("genres", [])


def discover(params: dict, token: str, key: str, pages: int = 2) -> list[dict]:
    """Retrieve a bounded pool; recommendation ranks THIS pool, not all TMDB titles."""
    movies: dict[int, dict] = {}
    for page in range(1, pages + 1):
        payload = api_get("discover/movie", {**params, "page": page}, _token=token, _api_key=key)
        for movie in payload.get("results", []):
            if movie.get("id") and movie.get("overview"):
                movies[movie["id"]] = movie
        if page >= min(payload.get("total_pages", 1), 500):
            break
    return list(movies.values())


def details(movie_id: int, token: str, key: str) -> dict:
    return api_get(f"movie/{int(movie_id)}", {"append_to_response": "credits,keywords,release_dates"}, _token=token, _api_key=key)


def search_person(name: str, token: str, key: str, department: str) -> dict | None:
    results = api_get("search/person", {"query": name.strip()}, _token=token, _api_key=key).get("results", [])
    if not results:
        return None
    desired = next((p for p in results if p.get("known_for_department") == department), None)
    return desired or results[0]


def search_movies(title: str, token: str, key: str) -> list[dict]:
    return api_get("search/movie", {"query": title.strip()}, _token=token, _api_key=key).get("results", [])


def similar_movie_candidates(movie_id: int, token: str, key: str) -> list[dict]:
    return api_get(f"movie/{movie_id}/similar", _token=token, _api_key=key).get("results", [])


def trending(token: str, key: str) -> list[dict]:
    return api_get("trending/movie/week", _token=token, _api_key=key).get("results", [])


def enrich(candidates: list[dict], token: str, key: str, limit: int = 24) -> list[dict]:
    """Fetch cast, directors, keywords, budget/revenue, certifications and runtime."""
    result = []
    for movie in candidates[:limit]:
        try:
            full = details(movie["id"], token, key)
            if movie.get("_topic_match"):
                full["_topic_match"] = True
            result.append(full)
        except TMDBError:
            # Individual removed/unavailable titles shouldn't crash the result screen.
            continue
    return result


def us_certification(movie: dict) -> str:
    """Select a nonempty US release certification when available."""
    for row in movie.get("release_dates", {}).get("results", []):
        if row.get("iso_3166_1") == "US":
            for release in row.get("release_dates", []):
                value = release.get("certification", "").strip()
                if value:
                    return value
    return "Not listed"


def poster_url(movie: dict) -> str | None:
    path = movie.get("poster_path")
    return f"{IMAGE_BASE}{path}" if path else None

    path = movie.get("poster_path")
    return f"{IMAGE_BASE}{path}" if path else None
