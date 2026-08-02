"""
TMDB Cast & Crew Enrichment Script
------------------------------------
Reads MovieLens movies.csv + links.csv, fetches cast & director data from
TMDB for each movie, and writes three normalized CSVs:

    person.csv  -> id, name, original_name, gender, popularity   (deduped)
    cast.csv    -> movie_id, person_id, character, order
    crew.csv    -> movie_id, person_id, job   (directors only)

Usage:
    export TMDB_API_KEY=your_key_here
    pip install aiohttp pandas
    python tmdb_enrich.py

Safe to interrupt (Ctrl+C) and re-run — it resumes from the last checkpoint.
"""

import os
import csv
import json
import time
import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from collections import deque

import aiohttp
import pandas as pd

# ---------------- Configuration ----------------
# NOTE: this must be your TMDB "API Read Access Token" (a long JWT string),
# not the shorter legacy v3 API key. Find it at:
# themoviedb.org -> Settings -> API -> API Read Access Token
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise EnvironmentError(
        "Set the TMDB_API_KEY environment variable before running this script.\n"
        "  export TMDB_API_KEY=your_read_access_token_here"
    )

BASE_URL = "https://api.themoviedb.org/3"
AUTH_HEADERS = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "accept": "application/json",
}
RATE_LIMIT_PER_SECOND = 40     # TMDB's soft ceiling
CONCURRENT_REQUESTS = 20       # keep below the rate limit ceiling with headroom
CHECKPOINT_EVERY = 50          # flush + save progress every N movies

DATA_DIR = Path("data")
MOVIES_CSV = DATA_DIR / "movies.csv"       # MovieLens: movieId,title,genres
LINKS_CSV = DATA_DIR / "links.csv"         # MovieLens: movieId,imdbId,tmdbId

PERSON_CSV = DATA_DIR / "person.csv"
CAST_CSV = DATA_DIR / "cast.csv"
CREW_CSV = DATA_DIR / "crew.csv"
CHECKPOINT_FILE = DATA_DIR / "_tmdb_checkpoint.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tmdb_enrich")


# ---------------- Rate limiter (sliding window) ----------------
class RateLimiter:
    """Caps total requests to `rate` per `period` seconds across all coroutines."""

    def __init__(self, rate: int, period: float = 1.0):
        self.rate = rate
        self.period = period
        self._timestamps = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > self.period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.rate:
                sleep_for = self.period - (now - self._timestamps[0])
                await asyncio.sleep(max(sleep_for, 0))
            self._timestamps.append(time.monotonic())


rate_limiter = RateLimiter(rate=RATE_LIMIT_PER_SECOND, period=1.0)


# ---------------- Checkpointing ----------------
def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT_FILE.write_text(json.dumps(list(done)))


# ---------------- CSV stores ----------------
class PersonStore:
    """Deduped append-only writer for person.csv, keyed by TMDB person id."""

    FIELDS = ["id", "name", "original_name", "gender", "popularity"]

    def __init__(self, path: Path):
        self.path = path
        self.known_ids = set()
        self._load_existing()
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._fh.flush()

    def _load_existing(self):
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.known_ids.add(int(row["id"]))
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()

    def add(self, person: dict):
        pid = person["id"]
        if pid in self.known_ids:
            return
        self.known_ids.add(pid)
        self._writer.writerow({
            "id": pid,
            "name": person.get("name", ""),
            "original_name": person.get("original_name", ""),
            "gender": person.get("gender", ""),
            "popularity": person.get("popularity", ""),
        })

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()


class RelationStore:
    """Generic append-only writer for cast.csv / crew.csv."""

    def __init__(self, path: Path, fields: list):
        self.path = path
        is_new = not path.exists() or path.stat().st_size == 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def add(self, row: dict):
        self._writer.writerow(row)

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()


# ---------------- TMDB fetch ----------------
async def validate_token(session: aiohttp.ClientSession):
    """One-time check that the Bearer token actually works, before we burn
    thousands of requests on a bad token."""
    url = f"{BASE_URL}/authentication"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        data = await resp.json()
        if resp.status != 200 or not data.get("success"):
            raise RuntimeError(
                f"TMDB token validation failed (status {resp.status}): {data}\n"
                "Check that TMDB_API_KEY is your Read Access Token (JWT), not the v3 api_key."
            )
        log.info("TMDB token validated successfully.")


async def fetch_credits(session: aiohttp.ClientSession, tmdb_id: int):
    url = f"{BASE_URL}/movie/{tmdb_id}/credits"

    for attempt in range(4):
        await rate_limiter.acquire()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 404:
                    log.warning(f"tmdb_id={tmdb_id} not found (404)")
                    return None
                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 1))
                    log.warning(f"Rate limited, sleeping {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                log.warning(f"tmdb_id={tmdb_id} unexpected status {resp.status}")
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(f"tmdb_id={tmdb_id} request error: {e} (attempt {attempt + 1})")
            await asyncio.sleep(1.5 * (attempt + 1))

    log.error(f"tmdb_id={tmdb_id} failed after retries")
    return None


# ---------------- Worker ----------------
async def process_movie(session, semaphore, movie_id, tmdb_id, person_store, cast_store, crew_store):
    async with semaphore:
        data = await fetch_credits(session, tmdb_id)
    if not data:
        return

    for member in data.get("cast", []):
        person_store.add(member)
        cast_store.add({
            "movie_id": movie_id,
            "person_id": member["id"],
            "character": member.get("character", ""),
            "order": member.get("order", ""),
        })

    for member in data.get("crew", []):
        if member.get("job") != "Director":
            continue
        person_store.add(member)
        crew_store.add({
            "movie_id": movie_id,
            "person_id": member["id"],
            "job": member.get("job", ""),
        })


# ---------------- Main ----------------
async def main():
    movies = pd.read_csv(MOVIES_CSV)
    links = pd.read_csv(LINKS_CSV)
    merged = movies.merge(links, on="movieId", how="left").dropna(subset=["tmdbId"])
    merged["tmdbId"] = merged["tmdbId"].astype(int)

    done = load_checkpoint()
    todo = merged[~merged["movieId"].isin(done)]
    log.info(f"{len(done)} movies already done, {len(todo)} remaining out of {len(merged)}")

    person_store = PersonStore(PERSON_CSV)
    cast_store = RelationStore(CAST_CSV, ["movie_id", "person_id", "character", "order"])
    crew_store = RelationStore(CREW_CSV, ["movie_id", "person_id", "job"])

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    processed_since_save = 0

    async with aiohttp.ClientSession(headers=AUTH_HEADERS) as session:
        await validate_token(session)

        async def bound_task(movie_id, tmdb_id):
            await process_movie(session, semaphore, movie_id, tmdb_id,
                                 person_store, cast_store, crew_store)
            return movie_id

        tasks = [bound_task(int(row.movieId), int(row.tmdbId)) for row in todo.itertuples()]

        for coro in asyncio.as_completed(tasks):
            movie_id = await coro
            done.add(movie_id)
            processed_since_save += 1
            if processed_since_save >= CHECKPOINT_EVERY:
                save_checkpoint(done)
                person_store.flush()
                cast_store.flush()
                crew_store.flush()
                processed_since_save = 0
                log.info(f"Progress: {len(done)}/{len(merged)} movies enriched")

    save_checkpoint(done)
    person_store.close()
    cast_store.close()
    crew_store.close()
    log.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())