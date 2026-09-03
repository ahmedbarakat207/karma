
import time
from typing import List, Dict, Any, Optional

try:
    import sqlean as sqlite3
except ImportError:
    import sqlite3
import sqlite_vec

from src import config


class MemoryStore:

    def __init__(self, db_path: str = config.MEMORY_DB_PATH, dim: int = config.EMBED_DIM):
        self.dim = dim
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    kind TEXT,
                    text TEXT,
                    source TEXT
                )
            """)
            # Migration check: Ensure source column exists on existing databases
            cols = [col[1] for col in self.db.execute("PRAGMA table_info(memories)").fetchall()]
            if "source" not in cols:
                try:
                    self.db.execute("ALTER TABLE memories ADD COLUMN source TEXT")
                except Exception:
                    pass

            self.db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
                USING vec0(embedding float[{self.dim}])
            """)

    def add(self, text: str, embedding: List[float], kind: str = "episodic_summary",
            ts: Optional[float] = None, source: Optional[str] = None) -> int:
        ts = ts or time.time()
        serialized = sqlite_vec.serialize_float32(embedding)

        with self.db:
            cur = self.db.execute(
                "INSERT INTO memories (ts, kind, text, source) VALUES (?, ?, ?, ?)",
                (ts, kind, text, source),
            )
            row_id = cur.lastrowid
            self.db.execute(
                "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
                (row_id, serialized),
            )
        return row_id

    def query(self, embedding: List[float], k: int = 5, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        serialized = sqlite_vec.serialize_float32(embedding)
        if kind:
            fetch_k = max(k * 4, 20)
            rows = self.db.execute(
                """
                SELECT m.text, m.ts, m.kind, v.distance, m.source, m.id
                FROM vec_items v
                JOIN memories m ON m.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ? AND m.kind = ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (serialized, fetch_k, kind, k),
            ).fetchall()
        else:
            rows = self.db.execute(
                """
                SELECT m.text, m.ts, m.kind, v.distance, m.source, m.id
                FROM vec_items v
                JOIN memories m ON m.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (serialized, k),
            ).fetchall()

        return [
            {
                "text": r[0],
                "ts": r[1],
                "kind": r[2],
                "distance": float(r[3]),
                "source": r[4] if len(r) > 4 else None,
                "id": r[5] if len(r) > 5 else None,
            }
            for r in rows
        ]

    def delete_by_source(self, source: str) -> int:
        with self.db:
            rows = self.db.execute("SELECT id FROM memories WHERE source = ?", (source,)).fetchall()
            for (r_id,) in rows:
                self.db.execute("DELETE FROM memories WHERE id = ?", (r_id,))
                self.db.execute("DELETE FROM vec_items WHERE rowid = ?", (r_id,))
        return len(rows)

    def delete_by_kind(self, kind: str) -> int:
        with self.db:
            rows = self.db.execute("SELECT id FROM memories WHERE kind = ?", (kind,)).fetchall()
            for (r_id,) in rows:
                self.db.execute("DELETE FROM memories WHERE id = ?", (r_id,))
                self.db.execute("DELETE FROM vec_items WHERE rowid = ?", (r_id,))
        return len(rows)

    def list_sources(self, kind: Optional[str] = "document") -> List[Dict[str, Any]]:
        query = "SELECT source, COUNT(*) FROM memories WHERE source IS NOT NULL"
        params = []
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " GROUP BY source ORDER BY source"
        rows = self.db.execute(query, params).fetchall()
        return [{"source": r[0], "count": r[1]} for r in rows]

    def prune_memories(self, max_age_days: int = 60, max_total_records: int = 1500) -> int:
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
        row = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0
