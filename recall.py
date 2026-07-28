"""
Quick way to test/inspect the long-term memory store.

Usage:
    python recall.py "what did it notice this morning"
"""
import sys
import time

import config
from memory_store import MemoryStore
from sentence_transformers import SentenceTransformer


def main():
    if len(sys.argv) < 2:
        print('Usage: python recall.py "your query here"')
        return

    query = " ".join(sys.argv[1:])

    embedder = SentenceTransformer(config.EMBED_MODEL_NAME)
    store = MemoryStore()

    embedding = embedder.encode(query).tolist()
    results = store.query(embedding, k=5)

    if not results:
        print("No memories stored yet -- let it run and sleep at least once.")
        return

    print(f"\nTop matches for: {query!r}\n")
    for r in results:
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        print(f"  [{t}] (dist={r['distance']:.3f}) {r['text']}")


if __name__ == "__main__":
    main()
