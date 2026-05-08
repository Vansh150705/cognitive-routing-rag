"""
Phase 1: Vector-Based Persona Matching
--------------------------------------
Embeddings: sentence-transformers (local, no API key needed)
Vector store: FAISS IndexFlatIP — normalised vectors = cosine similarity.
"""

import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_model = SentenceTransformer("all-MiniLM-L6-v2")

BOT_PERSONAS = {
    "bot_a": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space "
        "exploration. I dismiss regulatory concerns."
    ),
    "bot_b": (
        "I believe late-stage capitalism and tech monopolies are destroying "
        "society. I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c": (
        "I strictly care about markets, interest rates, trading algorithms, "
        "and making money. I speak in finance jargon and view everything "
        "through the lens of ROI."
    ),
}


def get_embedding(text: str) -> np.ndarray:
    """Return a unit-normalised embedding vector for text."""
    vec = _model.encode(text, normalize_embeddings=True).astype("float32")
    return vec


def build_persona_index():
    bot_ids = list(BOT_PERSONAS.keys())
    embeddings = np.stack([get_embedding(BOT_PERSONAS[bid]) for bid in bot_ids])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[phase1] persona index built — {index.ntotal} vectors, dim={dim}")
    return index, bot_ids


def route_post_to_bots(post_content, threshold=0.40, index=None, bot_ids=None):
    if index is None or bot_ids is None:
        index, bot_ids = build_persona_index()
    post_vec = get_embedding(post_content).reshape(1, -1)
    scores, indices = index.search(post_vec, k=len(bot_ids))
    matched = []
    for score, idx in zip(scores[0], indices[0]):
        bot_id = bot_ids[idx]
        sim = float(score)
        print(f"  {bot_id}: cosine_similarity={sim:.4f}  (threshold={threshold})")
        if sim >= threshold:
            matched.append({"bot_id": bot_id, "similarity": round(sim, 4)})
    matched.sort(key=lambda x: x["similarity"], reverse=True)
    return matched


if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits $100k as the Fed signals a rate pivot — risk assets surging.",
        "Big Tech layoffs reveal the dark side of unchecked AI monopolies.",
    ]
    idx, ids = build_persona_index()
    for post in test_posts:
        print(f"\n[post] {post!r}")
        results = route_post_to_bots(post, threshold=0.40, index=idx, bot_ids=ids)
        if results:
            print(f"  → routed to: {[r['bot_id'] for r in results]}")
        else:
            print("  → no bots matched")