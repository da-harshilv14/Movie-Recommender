"""
app.py — Streamlit frontend for the hybrid movie recommender.

Run with: streamlit run app.py

Flow:
  1. User types a movie title (partial match ok).
  2. If multiple matches, user picks the right one from a dropdown.
  3. Button triggers the hybrid recommender, results shown as ticket-stub cards.
"""

import streamlit as st

from recommend import find_movie_by_title, _movieid_to_title
from hybrid import hybrid_recommend_fn

st.set_page_config(page_title="Reel Match", page_icon="🎟️", layout="centered")

# ---------------------------------------------------------------------------
# Theme: cinema marquee — deep maroon, gold marquee accent, cream text.
# Signature element: recommendations rendered as ticket stubs.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #1C0F12;
        --bg-card: #2A1218;
        --gold: #D4A544;
        --gold-soft: #E8C77A;
        --cream: #F2E8D8;
        --muted: #B99DA0;
        --divider: #4A2530;
    }

    .stApp {
        background: radial-gradient(ellipse at top, #241318 0%, var(--bg) 60%);
        color: var(--cream);
    }

    /* Marquee header */
    .marquee-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.4rem;
        letter-spacing: 0.12em;
        color: var(--gold-soft);
        text-align: center;
        text-shadow: 0 0 18px rgba(212, 165, 68, 0.35);
        margin-bottom: 0;
        line-height: 1;
    }

    .marquee-sub {
        font-family: 'Inter', sans-serif;
        font-style: italic;
        font-weight: 400;
        color: var(--muted);
        text-align: center;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 2.2rem;
    }

    /* Inputs */
    .stTextInput input {
        background-color: var(--bg-card) !important;
        color: var(--cream) !important;
        border: 1px solid var(--divider) !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px var(--gold) !important;
    }
    .stTextInput label, .stSelectbox label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--muted) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        border-color: var(--divider) !important;
        color: var(--cream) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Button styled like a ticket-counter button */
    .stButton button {
        background: linear-gradient(180deg, var(--gold-soft), var(--gold)) !important;
        color: #1C0F12 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.8rem !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1.2rem !important;
        transition: filter 0.15s ease;
    }
    .stButton button:hover {
        filter: brightness(1.08);
    }

    /* Results header */
    .results-heading {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--cream);
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        font-size: 1.05rem;
    }
    .results-heading span {
        color: var(--muted);
        font-weight: 400;
        font-style: italic;
    }

    /* Ticket stub result card */
    .ticket {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background-color: var(--bg-card);
        border: 1px dashed var(--divider);
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.55rem;
    }
    .ticket-num {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.5rem;
        color: var(--gold);
        min-width: 2rem;
        text-align: center;
    }
    .ticket-title {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: var(--cream);
        font-size: 0.95rem;
    }

    /* Reduce default streamlit top padding */
    .block-container {
        padding-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="marquee-title">REEL MATCH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="marquee-sub">Tell us a film you loved — we\'ll find what\'s playing next.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
query = st.text_input("Search for a movie", placeholder="e.g. Toy Story")

if query:
    matches = find_movie_by_title(query)

    if matches.empty:
        st.warning("No movies found matching that search. Try a different title or spelling.")
    else:
        titles = matches["title"].tolist()
        movie_ids = matches["movieId"].tolist()
        title_to_id = dict(zip(titles, movie_ids))

        selected_title = st.selectbox(f"{len(titles)} match(es) found — pick one", titles)

        if st.button("Get Recommendations"):
            selected_id = title_to_id[selected_title]

            with st.spinner("Rolling the film..."):
                rec_ids = hybrid_recommend_fn(selected_id, k=10)
                rec_titles = [_movieid_to_title.get(mid, f"Unknown movie ({mid})") for mid in rec_ids]

            st.markdown(
                f'<div class="results-heading">Now showing — similar to <span>{selected_title}</span></div>',
                unsafe_allow_html=True,
            )

            tickets_html = "".join(
                f'<div class="ticket"><div class="ticket-num">{i:02d}</div>'
                f'<div class="ticket-title">{title}</div></div>'
                for i, title in enumerate(rec_titles, 1)
            )
            st.markdown(tickets_html, unsafe_allow_html=True)