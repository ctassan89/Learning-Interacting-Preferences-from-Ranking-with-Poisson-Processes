"""Build the input text of every movie for the text encoders.

Three input variants, all built only from movie metadata (never from ratings):

    overview    overview + tagline (as in ``movies_clean.csv``); a movie without an
                overview falls back to its title, so every movie gets a non-empty text.
    meta_core   "Title: ... (year). Genres: .... Director: .... Cast: <top 3>.
                Keywords: <first 5>. Plot: <overview + tagline>"
    meta_rich   meta_core plus language, writers (top 3), production companies
                (top 3) and saga/collection.

The plot always comes last, so when an encoder truncates a long input it cuts the
end of the plot and keeps the metadata. Empty fields are left out instead of being
written as "unknown", so missing metadata does not create a shared token that pulls
movies together.

Movies are loaded with ``experiments.TextEmbeddings.data.load_movies``, so
the order and the split are exactly those of ``data/processed/catalogue.csv``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

INPUT_VARIANTS = ("overview", "meta_core", "meta_rich")

# How many items of each list field are kept (lists are in TMDb order: billing
# order for the cast, credit order for writers and companies).
TOP_CAST = 3
TOP_KEYWORDS = 5
TOP_DIRECTORS = 2
TOP_WRITERS = 3
TOP_COMPANIES = 3

# Readable names for the most frequent original languages; others keep their code.
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "ja": "Japanese",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "cn": "Cantonese",
    "zh": "Mandarin",
    "sv": "Swedish",
    "da": "Danish",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "fa": "Persian",
    "hi": "Hindi",
    "no": "Norwegian",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
}


def _as_list(value: Any) -> list[str]:
    """Accept a parsed list, a pipe-joined string or a missing value."""
    if value is None:
        return []
    if isinstance(value, float):  # NaN from pandas
        return []
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    return [str(item) for item in value if str(item)]


def _as_text(value: Any) -> str:
    if value is None or isinstance(value, float):
        return ""
    return str(value).strip()


def _plot(movie: Mapping[str, Any]) -> str:
    """Overview + tagline; ``text`` (load_movies) or ``overview_tagline``."""
    text = _as_text(movie.get("text", movie.get("overview_tagline")))
    return text


def _year(movie: Mapping[str, Any]) -> str:
    year = movie.get("year")
    try:
        return str(int(year))
    except (TypeError, ValueError):
        return ""


def _field(label: str, items: Sequence[str]) -> str:
    return f"{label}: {', '.join(items)}." if items else ""


def overview_text(movie: Mapping[str, Any]) -> str:
    """The ``overview`` variant: overview + tagline, or the title if empty."""
    return _plot(movie) or _as_text(movie.get("title"))


def metadata_text(movie: Mapping[str, Any], rich: bool = False) -> str:
    """The ``meta_core`` (``rich=False``) or ``meta_rich`` variant of one movie."""
    title, year = _as_text(movie.get("title")), _year(movie)
    parts = []
    if title:
        parts.append(f"Title: {title} ({year})." if year else f"Title: {title}.")
    parts.append(_field("Genres", _as_list(movie.get("genres"))))
    directors = _as_list(movie.get("directors"))[:TOP_DIRECTORS]
    parts.append(_field("Director" if len(directors) == 1 else "Directors", directors))
    parts.append(_field("Cast", _as_list(movie.get("cast"))[:TOP_CAST]))
    parts.append(_field("Keywords", _as_list(movie.get("keywords"))[:TOP_KEYWORDS]))
    if rich:
        code = _as_text(movie.get("language"))
        if code and code != "unknown":
            parts.append(f"Language: {LANGUAGE_NAMES.get(code, code)}.")
        parts.append(_field("Writers", _as_list(movie.get("writers"))[:TOP_WRITERS]))
        parts.append(
            _field("Companies", _as_list(movie.get("companies"))[:TOP_COMPANIES])
        )
        parts.append(_field("Collection", _as_list(movie.get("collection"))))
    plot = _plot(movie)
    if plot:
        parts.append(f"Plot: {plot}")  # last, so truncation only cuts the plot
    return " ".join(part for part in parts if part)


def build_text(movie: Mapping[str, Any], variant: str) -> str:
    """Input text of one movie (a dict or a pandas row) for the given variant."""
    if variant == "overview":
        return overview_text(movie)
    if variant == "meta_core":
        return metadata_text(movie, rich=False)
    if variant == "meta_rich":
        return metadata_text(movie, rich=True)
    raise ValueError(f"Unknown input variant {variant!r}; use one of {INPUT_VARIANTS}.")


def build_texts(movies: Any, variant: str) -> list[str]:
    """Input texts of every row of a DataFrame (or of a list of dicts)."""
    rows = movies.to_dict("records") if hasattr(movies, "to_dict") else movies
    return [build_text(movie, variant) for movie in rows]
