"""
Long-Term Memory Subsystem ("The Archive").
Single-file sqlite-vec vector database for semantic retrieval of episodic memories.
"""
import time
from typing import List, Dict, Any, Optional

try:
    import sqlean as sqlite3
except ImportError:
    import sqlite3
import sqlite_vec

from src import config


class MemoryStore:
    """Thread-safe vector store backed by SQLite and sqlite-vec."""

    def __init__(self, db_path: str = config.MEMORY_DB_PATH, dim: int = config.EMBED_DIM):
        self.dim = dim
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create relational and virtual vector tables if they do not exist."""
        with self.db:
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

    def add(self, text: str, embedding: List[float], kind: str = "episodic_summary",
            ts: Optional[float] = None) -> int:
        """Insert a memory and its corresponding vector embedding."""
        ts = ts or time.time()
        serialized = sqlite_vec.serialize_float32(embedding)

        with self.db:
            cur = self.db.execute(
                "INSERT INTO memories (ts, kind, text) VALUES (?, ?, ?)",
                (ts, kind, text),
            )
            row_id = cur.lastrowid
            self.db.execute(
                "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
                (row_id, serialized),
            )
        return row_id

    def query(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Perform K-Nearest Neighbors (KNN) search over memories."""
        serialized = sqlite_vec.serialize_float32(embedding)
        rows = self.db.execute(
            """
            SELECT m.text, m.ts, m.kind, v.distance
            FROM vec_items v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (serialized, k),
        ).fetchall()

        return [
            {"text": r[0], "ts": r[1], "kind": r[2], "distance": float(r[3])}
            for r in rows
        ]

    def prune_memories(self, max_age_days: int = 60, max_total_records: int = 1500) -> int:
        """Prunes memories older than cutoff or exceeding max total count."""
        cutoff_ts = time.time() - (max_age_days * 86400)
        pruned_count = 0

        with self.db:
            old_rows = self.db.execute(
                "SELECT id FROM memories WHERE ts < ?", (cutoff_ts,)
            ).fetchall()

            for (r_id,) in old_rows:
                self.db.execute("DELETE FROM memories WHERE id = ?", (r_id,))
                self.db.execute("DELETE FROM vec_items WHERE rowid = ?", (r_id,))
                pruned_count += 1

            count = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count > max_total_records:
                excess = count - max_total_records
                excess_rows = self.db.execute(
                    "SELECT id FROM memories ORDER BY ts ASC LIMIT ?", (excess,)
                ).fetchall()
                for (r_id,) in excess_rows:
                    self.db.execute("DELETE FROM memories WHERE id = ?", (r_id,))
                    self.db.execute("DELETE FROM vec_items WHERE rowid = ?", (r_id,))
                    pruned_count += 1

        if pruned_count > 0:
            print(f"[memory_store] pruned {pruned_count} old memory entries.")
        return pruned_count

    def count(self) -> int:
        """Return total number of memories stored."""
        row = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0
