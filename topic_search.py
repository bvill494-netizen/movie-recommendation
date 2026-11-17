
"""Extract story topics from natural-language queries and discover topic-matched TMDB titles."""
import re

from tmdb_client import api_get, discover

STOPWORDS = set("i me my we our you your want need please find show recommend looking for that have has with without of a an the some something good best great films film movies movie about on involving featuring set in where and or to from like based similar starring set".split())
# Words that are preferences rather than the movie's subject.
MOOD_WORDS = {"dark", "uplifting", "happy", "sad", "funny", "scary", "serious", "lighthearted", "exciting"}
GENRE_WORDS = set("action comedy drama horror romance romantic sci-fi scifi science fiction thriller documentary animation animated fantasy crime mystery western family adventure".split())


def topic_terms(text: str) -> list[str]:
    """Ignore search instructions, ratings, years, and genre labels when matching a plot topic."""
    cleaned = text.lower().replace("sci-fi", "scifi")
    cleaned = re.sub(r"\b(?:rated?|rating)\s*(?:at least|above|over|>=)?\s*\d+(?:\.\d+)?\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:after|before|from|since|until)\s+(?:19|20)\d{2}\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:19|20)\d{2}s?\b", " ", cleaned)
    words = re.findall(r"[a-z]+\d*|\d+", cleaned)
    return [w for w in words if len(w) > 1 and w not in STOPWORDS and w not in MOOD_WORDS and w not in GENRE_WORDS]


def topic_queries(query: str) -> list[str]:
    """Generate a small number of targeted TMDB keyword searches."""
    words = topic_terms(query)
    if not words:
        return []
    topic = " ".join(words)
    motorsport = bool(re.search(r"\b(race|races|racing|racer|f1|nascar|motorsport|formula|grand prix)\b", topic))
    cars = bool(re.search(r"\b(car|cars|racecar|vehicle|automobile|driving)\b", topic))
    if motorsport and cars:
        return ["car racing", "street racing", "automobile racing", "motor racing", "race car", "car race"]
    if motorsport:
        return ["racing", "motor racing", "motorsport", "formula 1"]
    phrases = [topic]
    if len(words) > 1:
        phrases += [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
    phrases += sorted(set(words), key=words.index)
    return list(dict.fromkeys(phrases))[:5]


def keyword_ids(query: str, token: str, key: str) -> list[int]:
    """Lookup exact TMDB topic labels; do not accept unrelated fuzzy search results."""
    result_ids = []
    for phrase in topic_queries(query):
        payload = api_get("search/keyword", {"query": phrase}, _token=token, _api_key=key)
        exact = [row for row in payload.get("results", []) if row.get("name", "").casefold() == phrase.casefold()]
        for row in exact[:2]:
            if row.get("id") not in result_ids:
                result_ids.append(row["id"])
    return result_ids[:8]


def discover_for_query(params: dict, query: str, token: str, key: str) -> list[dict]:
    """First retrieve films about the topic, rather than blindly ranking popular films.

    All original discover filters are applied even when searching by topic.
    """
    if not topic_terms(query):
        return discover(params, token, key, pages=2)
    ids = keyword_ids(query, token, key)
    if ids:
        narrowed = dict(params)
        narrowed["with_keywords"] = "|".join(str(x) for x in ids)
        matches = discover(narrowed, token, key, pages=3)
        if matches:
            for movie in matches:
                movie["_topic_match"] = True
            return matches
    # If TMDB has no matching keyword entries, examine a larger filtered pool.
    # The ranking function will still reject stories unrelated to the query.
    return discover(params, token, key, pages=5)
