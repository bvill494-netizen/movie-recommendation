
"""Two complementary recommendation approaches: TF-IDF and lexicon sentiment."""
from __future__ import annotations

import math
import re

from topic_search import topic_terms
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def genre_names(movie: dict) -> list[str]:
    return [x["name"] for x in movie.get("genres", []) if x.get("name")]


def director_names(movie: dict) -> list[str]:
    return [p["name"] for p in movie.get("credits", {}).get("crew", []) if p.get("job") == "Director"][:2]


def cast_names(movie: dict) -> list[str]:
    return [p["name"] for p in movie.get("credits", {}).get("cast", [])[:5]]


def keyword_names(movie: dict) -> list[str]:
    block = movie.get("keywords", {})
    items = block.get("keywords", block.get("results", []))
    return [x["name"] for x in items if x.get("name")]


def movie_text(movie: dict) -> str:
    """A unified text representation of attributes and plot for TF-IDF."""
    return " ".join([
        movie.get("title") or "", movie.get("overview") or "",
        " ".join(genre_names(movie) * 3),
        " ".join(keyword_names(movie) * 2),
        " ".join(cast_names(movie)), " ".join(director_names(movie)),
    ]).strip()


def polarity(movie: dict) -> float:
    """TextBlob polarity ∈ [-1,1]. Plot wording is only an APPROXIMATION of mood."""
    return float(TextBlob(movie.get("overview") or "").sentiment.polarity)


def _quality(movie: dict) -> float:
    votes = max(0, int(movie.get("vote_count") or 0))
    rating = max(0.0, min(10.0, float(movie.get("vote_average") or 0)))
    confidence = min(1.0, math.log1p(votes) / math.log1p(1000))
    return rating / 10 * confidence


def _topic_evidence(movie: dict, terms: list[str]) -> float:
    """Hard relevance gate: do not recommend an unrelated film just because it is popular."""
    if not terms:
        return 1.0
    if movie.get("_topic_match"):
        return 1.0  # Matched an exact topic keyword in TMDB discover.
    text = movie_text(movie).lower()
    cleaned_terms = set(terms)
    racing = bool(cleaned_terms.intersection({"race", "races", "racing", "racer", "motorsport", "nascar", "formula"}))
    cars = bool(cleaned_terms.intersection({"car", "cars", "racecar", "vehicle", "automobile", "driving"}))
    if racing and cars:
        # A generic mention of "racing against time" isn't enough.
        race_hit = bool(re.search(r"\b(race|races|racer|racers|racing|nascar|motorsport|formula.?one|grand prix)\b", text))
        car_hit = bool(re.search(r"\b(car|cars|vehicle|automobile|racecar|driver|driving|motor)\b", text))
        return 0.8 if race_hit and car_hit else 0.0
    coverage = sum(bool(re.search(r"\b" + re.escape(t) + r"(?:s|es|ing)?\b", text)) for t in set(terms))
    required = max(1, math.ceil(len(set(terms)) * 0.6))
    return coverage / len(set(terms)) if coverage >= required else 0.0


def rank_movies(movies: list[dict], query: str = "", mood: float = 0.5,
                liked_movies: list[dict] | None = None, exclude_ids: set[int] | None = None) -> list[dict]:
    """Rank a bounded TMDB candidate pool; never train on nonexistent user ratings."""
    liked_movies = liked_movies or []
    exclude_ids = exclude_ids or set()
    pool = [m for m in movies if m.get("id") not in exclude_ids and m.get("overview")]
    if not pool:
        return []
    docs = [movie_text(m) for m in pool]
    profile = " ".join(movie_text(m) for m in liked_movies[:12])
    target_text = (query + " " + profile).strip()
    content = [0.0] * len(pool)
    if target_text:
        try:
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000).fit_transform(docs + [target_text])
            content = cosine_similarity(matrix[:-1], matrix[-1]).ravel().tolist()
        except ValueError:
            # Empty vocabulary, e.g. stopwords-only user query.
            pass
    liked_genres = set().union(*(set(genre_names(m)) for m in liked_movies)) if liked_movies else set()
    result = []
    topic = topic_terms(query) if query.strip() else []
    for movie, similarity in zip(pool, content):
        evidence = _topic_evidence(movie, topic)
        if topic and evidence == 0:
            continue
        sent = polarity(movie)
        mood_match = 1 - abs(sent - (mood * 2 - 1)) / 2
        quality = _quality(movie)
        # 2 NLP approaches explicitly blended with a confidence-adjusted rating.
        score = (0.65 * similarity + 0.18 * evidence + 0.09 * mood_match + 0.08 * quality) if topic else (0.58 * similarity + 0.22 * mood_match + 0.20 * quality)
        common_genres = liked_genres.intersection(genre_names(movie))
        explanations = []
        if topic and movie.get("_topic_match"):
            explanations.append("Matched your topic through TMDB movie keywords")
        elif topic:
            explanations.append("Topic terms appear in this movie description or metadata")
        if common_genres:
            explanations.append("Shared genres: " + ", ".join(sorted(common_genres)[:3]))
        if target_text and similarity > 0.02:
            explanations.append("Similarity in plot, keywords or people")
        explanations.append("Plot language sounds " + ("upbeat" if sent > 0.15 else "somber" if sent < -0.15 else "neutral"))
        row = dict(movie)
        row["_score"] = score
        row["_content"] = float(similarity)
        row["_sentiment"] = sent
        row["_why"] = "; ".join(explanations)
        result.append(row)
    return sorted(result, key=lambda m: m["_score"], reverse=True)
