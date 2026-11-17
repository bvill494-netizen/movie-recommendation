
"""Small, explainable natural-language filter parser (not a full chat model)."""
import re

ALIASES = {"sci-fi": "Science Fiction", "scifi": "Science Fiction",
           "science fiction": "Science Fiction", "romcom": "Romance",
           "romantic": "Romance", "funny": "Comedy", "scary": "Horror",
           "thriller": "Thriller", "action": "Action", "comedy": "Comedy",
           "horror": "Horror", "animation": "Animation", "documentary": "Documentary",
           "drama": "Drama", "fantasy": "Fantasy", "mystery": "Mystery",
           "western": "Western", "romance": "Romance", "crime": "Crime"}


def parse_query(text: str) -> dict:
    lower = text.lower()
    found = {"genres": [], "year_min": None, "year_max": None, "rating_min": None}
    for alias, canonical in ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lower) and canonical not in found["genres"]:
            found["genres"].append(canonical)
    decade = re.search(r"\b(19\d0|20\d0)s\b", lower)
    if decade:
        y = int(decade.group(1))
        found["year_min"], found["year_max"] = y, y + 9
    after = re.search(r"\b(?:after|since|from)\s+(19\d{2}|20\d{2})\b", lower)
    before = re.search(r"\b(?:before|until)\s+(19\d{2}|20\d{2})\b", lower)
    if after:
        found["year_min"] = int(after.group(1)) + (1 if "after" in after.group(0) else 0)
    if before:
        found["year_max"] = int(before.group(1)) - (1 if "before" in before.group(0) else 0)
    rating = re.search(r"\b(?:rated?|rating)\s*(?:at least|above|over|>=)?\s*(\d{1,2}(?:\.\d)?)\b", lower)
    if rating:
        value = float(rating.group(1))
        if 0 <= value <= 10:
            found["rating_min"] = value
    return found
