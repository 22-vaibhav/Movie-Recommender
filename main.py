import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv

# ENV
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# ================================
# LIFESPAN (startup + shutdown)
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading resources...")

    # Load pickles
    with open(DF_PATH, "rb") as f:
        app.state.df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        app.state.indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        app.state.tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        app.state.tfidf_obj = pickle.load(f)

    # Build mapping
    app.state.TITLE_TO_IDX = build_title_to_idx_map(app.state.indices_obj)

    if app.state.df is None or "title" not in app.state.df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    # Reusable HTTP client
    app.state.client = httpx.AsyncClient(timeout=20)

    print("✅ All resources loaded")

    yield

    print("🛑 Shutting down...")
    await app.state.client.aclose()


# FASTAPI APP
app = FastAPI(
    title="Movie Recommender API",
    version="3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# MODELS
# ================================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# ================================
# UTILS
# ================================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    return f"{TMDB_IMG_500}{path}" if path else None


async def tmdb_get(path: str, params: Dict[str, Any], request: Request) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        client = request.app.state.client
        r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(502, f"TMDB request error: {repr(e)}")

    if r.status_code != 200:
        raise HTTPException(502, f"TMDB error {r.status_code}: {r.text}")

    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    return [
        TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
        for m in (results or [])[:limit]
    ]


async def tmdb_movie_details(movie_id: int, request: Request) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"}, request)
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, request: Request, page: int = 1):
    return await tmdb_get("/search/movie", {
        "query": query,
        "include_adult": "false",
        "language": "en-US",
        "page": page,
    }, request)


async def tmdb_search_first(query: str, request: Request):
    data = await tmdb_search_movies(query, request)
    results = data.get("results", [])
    return results[0] if results else None


# ================================
# TF-IDF HELPERS
# ================================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx = {}
    for k, v in indices.items():
        title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx


def get_local_idx_by_title(title: str, request: Request) -> int:
    TITLE_TO_IDX = request.app.state.TITLE_TO_IDX
    key = _norm_title(title)

    if key in TITLE_TO_IDX:
        return TITLE_TO_IDX[key]

    raise HTTPException(404, f"Title not found: '{title}'")


def tfidf_recommend_titles(query_title: str, request: Request, top_n: int = 10):
    df = request.app.state.df
    tfidf_matrix = request.app.state.tfidf_matrix

    idx = get_local_idx_by_title(query_title, request)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    results = []
    for i in order:
        if i == idx:
            continue
        results.append((str(df.iloc[i]["title"]), float(scores[i])))
        if len(results) >= top_n:
            break

    return results


async def attach_tmdb_card_by_title(title: str, request: Request):
    try:
        m = await tmdb_search_first(title, request)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except:
        return None


# ================================
# ROUTES
# ================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(request: Request, category: str = "popular", limit: int = 24):
    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {"language": "en-US"}, request)
    else:
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US"}, request)

    return await tmdb_cards_from_results(data.get("results", []), limit)


@app.get("/tmdb/search")
async def tmdb_search(request: Request, query: str, page: int = 1):
    return await tmdb_search_movies(query, request, page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(request: Request, tmdb_id: int):
    return await tmdb_movie_details(tmdb_id, request)


@app.get("/recommend/tfidf")
async def recommend_tfidf(request: Request, title: str, top_n: int = 10):
    recs = tfidf_recommend_titles(title, request, top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(request: Request, query: str):
    best = await tmdb_search_first(query, request)
    if not best:
        raise HTTPException(404, f"No movie found for query: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id, request)

    recs = tfidf_recommend_titles(details.title, request)

    tfidf_items = []
    for t, s in recs:
        card = await attach_tmdb_card_by_title(t, request)
        tfidf_items.append(TFIDFRecItem(title=t, score=s, tmdb=card))

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=[]
    )