import os
from pathlib import Path

import faiss
import numpy as np
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "db" / "semantic.index"
EMBED_DIM = 1536

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:
        raise RuntimeError("OpenAI client not installed.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    _client = OpenAI()
    return _client

def _load_index():
    if INDEX_PATH.exists():
        try:
            return faiss.read_index(str(INDEX_PATH))
        except Exception:
            pass
    return faiss.IndexFlatL2(EMBED_DIM)

index = _load_index()

def embed(text):
    client = _get_client()
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(emb.data[0].embedding).astype("float32")

def store_email(text):
    try:
        index.add(embed(text).reshape(1, -1))
    except Exception:
        return
    try:
        faiss.write_index(index, str(INDEX_PATH))
    except Exception:
        return

def get_similar_emails(text):
    if index.ntotal == 0:
        return "None"

    try:
        D, I = index.search(embed(text).reshape(1, -1), 3)
    except Exception:
        return "None"
    return f"Found {len(I[0])} similar emails"
