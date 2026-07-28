"""
Long-term memory: a single-file sqlite-vec database. This is the "rag file" --
everything consolidation produces at sleep time lands here, and recall.py
(or the wake-time context loader) queries it back by similarity.
"""
try:
    import sqlean as sqlite3
except ImportError:
    import sqlite3
import time
import sqlite_vec

import config


class MemoryStore:
    def __init__(self, db_path=config.MEMORY_DB_PATH, dim=config.EMBED_DIM):
        self.dim = dim
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                kind TEXT,
                text TEXT
            )
        """)
        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
            USING vec0(embedding float[{self.dim}])
        """)
        self.db.commit()

    def add(self, text, embedding, kind="episodic_summary", ts=None):
        ts = ts or time.time()
        cur = self.db.execute(
            "INSERT INTO memories (ts, kind, text) VALUES (?, ?, ?)",
            (ts, kind, text),
        )
        row_id = cur.lastrowid
        self.db.execute(
            "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
            (row_id, sqlite_vec.serialize_float32(embedding)),
        )
        self.db.commit()
        return row_id

    def query(self, embedding, k=5):
        rows = self.db.execute(
            """
            SELECT m.text, m.ts, m.kind, v.distance
            FROM vec_items v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ?
            ORDER BY v.distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(embedding), k),
        ).fetchall()
        return [
            {"text": r[0], "ts": r[1], "kind": r[2], "distance": r[3]} for r in rows
        ]

    def prune_memories(self, max_age_days=60, max_total_records=1500):
        """Prunes old or low-importance memories to keep long-term database lean."""
        cutoff_ts = time.time() - (max_age_days * 86400)
        # Delete entries older than cutoff
        old_rows = self.db.execute(
            "SELECT id FROM memories WHERE ts < ?", (cutoff_ts,)
        ).fetchall()
        
        for (r_id,) in old_rows:
            self.db.execute("DELETE FROM memories WHERE id = ?", (r_id,))
            self.db.execute("DELETE FROM vec_items WHERE rowid = ?", (r_id,))
        
        # Enforce max record ceiling
        count = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > max_total_records:
            excess = count - max_total_records
            excess_rows = self.db.execute(
                "SELECT id FROM memories ORDER BY ts ASC LIMIT ?", (excess,)
            ).fetchall()
            for (r_id,) in excess_rows:
                self.db.execute("DELETE FROM memories WHERE id = ?", (r_id,))
                self.db.execute("DELETE FROM vec_items WHERE rowid = ?", (r_id,))
        
        self.db.commit()
        print(f"[memory_store] pruned memory db (cleaned {len(old_rows)} old entries).")

