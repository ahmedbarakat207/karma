#!/usr/bin/env python3
import sys
import time

from sentence_transformers import SentenceTransformer

from src import config
from src.memory.store import MemoryStore


def main():
    if len(sys.argv) < 2:
        print('Usage: python recall.py "your query here"')
        return

    query = " ".join(sys.argv[1:])

    embedder = SentenceTransformer(getattr(config, "EMBED_MODEL_PATH", config.EMBED_MODEL_NAME))
    store = MemoryStore()

    embedding = embedder.encode(query).tolist()
    results = store.query(embedding, k=5)

    if not results:
        print("No memories stored yet -- run Karma and trigger sleep consolidation at least once.")
        return

    print(f"\nTop matches for: {query!r}\n")
    for r in results:
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        print(f"  [{t}] (dist={r['distance']:.3f}) {r['text']}")


if __name__ == "__main__":
    main()
