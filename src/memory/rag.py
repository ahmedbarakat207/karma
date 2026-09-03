import argparse
import os
import re
import sys
from typing import List, Dict, Any, Optional

try:
    from markitdown import MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:
    _MARKITDOWN_AVAILABLE = False

from src import config
from src.memory.store import MemoryStore


class DocumentRAG:
    def __init__(self, store: Optional[MemoryStore] = None, embedder: Optional[Any] = None,
                 chunk_size: int = 450, chunk_overlap: int = 80):
        self.store = store or MemoryStore()
        self._embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._md_converter: Optional[Any] = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            path = getattr(config, "EMBED_MODEL_PATH", config.EMBED_MODEL_NAME)
            self._embedder = SentenceTransformer(path)
        return self._embedder

    @property
    def md(self):
        if self._md_converter is None:
            if not _MARKITDOWN_AVAILABLE:
                raise RuntimeError("markitdown not installed. Run: pip install 'markitdown[pdf]'")
            self._md_converter = MarkItDown()
        return self._md_converter

    def parse_pdf(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        result = self.md.convert(file_path)
        return (getattr(result, "text_content", "") or "").strip()

    def chunk_markdown(self, text: str, source: str) -> List[str]:
        if not text:
            return []

        raw_blocks = re.split(r'\n(?=#{1,6}\s)', text)
        chunks: List[str] = []
        current: List[str] = []
        cur_len = 0
        header = ""

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            hm = re.match(r'^(#{1,6}\s+[^\n]+)', block)
            if hm:
                header = hm.group(1).strip('# ').strip()

            for p in block.split('\n\n'):
                p = p.strip()
                if not p:
                    continue

                if cur_len + len(p) > self.chunk_size and current:
                    prefix = f"[Document: {source}" + (f" | Section: {header}]" if header else "]")
                    chunks.append(f"{prefix}\n{chr(10).join(current)}")
                    if len(current[-1]) < self.chunk_overlap:
                        current = [current[-1], p]
                        cur_len = len(current[0]) + len(p)
                    else:
                        current = [p]
                        cur_len = len(p)
                else:
                    current.append(p)
                    cur_len += len(p)

        if current:
            prefix = f"[Document: {source}" + (f" | Section: {header}]" if header else "]")
            chunks.append(f"{prefix}\n{chr(10).join(current)}")

        return chunks

    def ingest_pdf(self, file_path: str, verbose: bool = True) -> int:
        path = os.path.abspath(file_path)
        name = os.path.basename(path)

        if verbose:
            print(f"[rag] parsing '{name}'...")

        md_text = self.parse_pdf(path)
        if not md_text:
            if verbose:
                print(f"[rag] warning: '{name}' produced no text")
            return 0

        chunks = self.chunk_markdown(md_text, name)
        if not chunks:
            return 0

        deleted = self.store.delete_by_source(name)
        if deleted > 0 and verbose:
            print(f"[rag] replaced {deleted} old chunks for '{name}'")

        if verbose:
            print(f"[rag] embedding {len(chunks)} chunks...")

        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        for chunk, emb in zip(chunks, embeddings):
            self.store.add(text=chunk, embedding=emb, kind="document", source=name)

        if verbose:
            print(f"[rag] done: '{name}' ({len(chunks)} chunks)")

        return len(chunks)

    def ingest_directory(self, dir_path: str, extensions: tuple = (".pdf", ".txt", ".md")) -> Dict[str, int]:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = {}
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                if any(f.lower().endswith(ext) for ext in extensions):
                    try:
                        results[f] = self.ingest_pdf(os.path.join(root, f))
                    except Exception as e:
                        print(f"[rag] error on '{f}': {e}")
                        results[f] = 0
        return results

    def retrieve(self, query: str, k: int = 3, threshold: float = 1.28) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        emb = self.embedder.encode(query).tolist()
        hits = self.store.query(emb, k=k, kind="document")
        return [h for h in hits if h["distance"] <= threshold]

    def get_rag_context(self, query: str, k: int = 3, threshold: float = 1.28) -> str:
        hits = self.retrieve(query, k=k, threshold=threshold)
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            relevance = 1.0 / (1.0 + h["distance"])
            parts.append(f"--- Excerpt {i} (Relevance: {relevance:.2f}) ---\n{h['text'].strip()}")
        return "\n\n".join(parts)

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.store.list_sources(kind="document")

    def clear_documents(self) -> int:
        return self.store.delete_by_kind("document")


def retrieve_document_context(query: str, store=None, embedder=None, k: int = 2) -> str:
    if not store or not embedder or not query.strip():
        return ""
    try:
        return DocumentRAG(store=store, embedder=embedder).get_rag_context(query, k=k)
    except Exception as e:
        config.log_debug(f"[rag] retrieval error: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Karma Document RAG CLI")
    parser.add_argument("--ingest", "-i", type=str, help="PDF or document file to ingest")
    parser.add_argument("--dir", "-d", type=str, help="Directory of documents to ingest")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--list", "-l", action="store_true", help="List indexed documents")
    parser.add_argument("--clear", action="store_true", help="Clear all indexed documents")
    parser.add_argument("--k", type=int, default=3, help="Chunks to retrieve (default: 3)")

    args = parser.parse_args()
    rag = DocumentRAG()

    if args.clear:
        print(f"Cleared {rag.clear_documents()} chunks.")
        return

    if args.list:
        sources = rag.list_documents()
        if not sources:
            print("No documents indexed.")
        else:
            for s in sources:
                print(f"  {s['source']}: {s['count']} chunks")
        return

    if args.ingest:
        rag.ingest_pdf(args.ingest)
        return

    if args.dir:
        rag.ingest_directory(args.dir)
        return

    if args.query:
        ctx = rag.get_rag_context(args.query, k=args.k)
        print(ctx if ctx else "No relevant docs found.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
