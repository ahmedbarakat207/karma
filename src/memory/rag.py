"""
Document Retrieval-Augmented Generation (RAG) Subsystem.
Converts PDF documents to Markdown via Microsoft MarkItDown,
chunks semantically, embeds using SentenceTransformer, and stores
into the persistent sqlite-vec database.
"""
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
    """End-to-end RAG manager for PDF and document ingestion, indexing, and retrieval."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        embedder: Optional[Any] = None,
        chunk_size: int = 450,
        chunk_overlap: int = 80,
    ):
        self.store = store or MemoryStore()
        self._embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._md_converter: Optional[Any] = None

    @property
    def embedder(self):
        """Lazy-load the SentenceTransformer embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            embed_path = getattr(config, "EMBED_MODEL_PATH", config.EMBED_MODEL_NAME)
            self._embedder = SentenceTransformer(embed_path)
        return self._embedder

    @property
    def md(self):
        """Lazy-load the MarkItDown converter instance."""
        if self._md_converter is None:
            if not _MARKITDOWN_AVAILABLE:
                raise RuntimeError(
                    "markitdown library is not installed. Install via: pip install 'markitdown[pdf]'"
                )
            self._md_converter = MarkItDown()
        return self._md_converter

    def parse_pdf(self, file_path: str) -> str:
        """Converts a PDF file to clean Markdown text using MarkItDown."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        result = self.md.convert(file_path)
        markdown_text = getattr(result, "text_content", "") or ""
        return markdown_text.strip()

    def chunk_markdown(self, markdown_text: str, source_name: str) -> List[str]:
        """
        Semantically chunks Markdown text respecting headers, lists, and paragraphs.
        Prefixes each chunk with source metadata for accurate LLM grounding.
        """
        if not markdown_text:
            return []

        # Split into blocks by double newlines or markdown headers
        raw_blocks = re.split(r'\n(?=#{1,6}\s)', markdown_text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0
        current_header = ""

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            # Detect header if block starts with #
            header_match = re.match(r'^(#{1,6}\s+[^\n]+)', block)
            if header_match:
                current_header = header_match.group(1).strip('# ').strip()

            # Split large blocks into paragraphs if needed
            paragraphs = block.split('\n\n')
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                p_len = len(p)
                if current_len + p_len > self.chunk_size and current_chunk:
                    # Finalize chunk
                    chunk_body = "\n\n".join(current_chunk)
                    prefix = f"[Document: {source_name}" + (f" | Section: {current_header}]" if current_header else "]")
                    chunks.append(f"{prefix}\n{chunk_body}")

                    # Keep last paragraph for overlap if within budget
                    if len(current_chunk[-1]) < self.chunk_overlap:
                        current_chunk = [current_chunk[-1], p]
                        current_len = len(current_chunk[0]) + p_len
                    else:
                        current_chunk = [p]
                        current_len = p_len
                else:
                    current_chunk.append(p)
                    current_len += p_len

        if current_chunk:
            chunk_body = "\n\n".join(current_chunk)
            prefix = f"[Document: {source_name}" + (f" | Section: {current_header}]" if current_header else "]")
            chunks.append(f"{prefix}\n{chunk_body}")

        return chunks

    def ingest_pdf(self, file_path: str, verbose: bool = True) -> int:
        """
        Parses a PDF with MarkItDown, chunks it, embeds each chunk, and saves to sqlite-vec.
        Replaces any previously indexed chunks for the same filename to avoid duplicates.
        """
        abs_path = os.path.abspath(file_path)
        source_name = os.path.basename(abs_path)

        if verbose:
            print(f"[rag] 📄 Parsing '{source_name}' with MarkItDown...")

        markdown_text = self.parse_pdf(abs_path)
        if not markdown_text:
            if verbose:
                print(f"[rag] ⚠️ Warning: '{source_name}' produced no extractable text.")
            return 0

        chunks = self.chunk_markdown(markdown_text, source_name)
        if not chunks:
            return 0

        # Remove existing chunks for this source to ensure idempotency
        deleted = self.store.delete_by_source(source_name)
        if deleted > 0 and verbose:
            print(f"[rag] 🔄 Replaced {deleted} existing chunks for '{source_name}'.")

        if verbose:
            print(f"[rag] 🧠 Embedding {len(chunks)} chunks using SentenceTransformer...")

        # Batch compute embeddings for high throughput
        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()

        for chunk_text, emb in zip(chunks, embeddings):
            self.store.add(
                text=chunk_text,
                embedding=emb,
                kind="document",
                source=source_name,
            )

        if verbose:
            print(f"[rag] ✓ Successfully indexed '{source_name}' ({len(chunks)} semantic chunks).")

        return len(chunks)

    def ingest_directory(self, dir_path: str, extensions: tuple = (".pdf", ".txt", ".md")) -> Dict[str, int]:
        """Scans a directory and indexes all matching documents."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        results = {}
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                if any(f.lower().endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, f)
                    try:
                        count = self.ingest_pdf(full_path, verbose=True)
                        results[f] = count
                    except Exception as e:
                        print(f"[rag] ⚠️ Error indexing '{f}': {e}")
                        results[f] = 0
        return results

    def retrieve(self, query: str, k: int = 3, threshold: float = 1.35) -> List[Dict[str, Any]]:
        """
        Retrieves top-k document chunks relevant to the user query.
        Filters out low-confidence hits where vector distance exceeds threshold.
        """
        if not query.strip():
            return []

        emb = self.embedder.encode(query).tolist()
        hits = self.store.query(emb, k=k, kind="document")

        # Filter by distance threshold
        relevant = [h for h in hits if h["distance"] <= threshold]
        return relevant

    def get_rag_context(self, query: str, k: int = 3, threshold: float = 1.35) -> str:
        """
        Returns a formatted markdown string of relevant document passages
        ready to be injected into LLM cognitive prompt.
        """
        hits = self.retrieve(query, k=k, threshold=threshold)
        if not hits:
            return ""

        context_lines = []
        for i, hit in enumerate(hits, 1):
            text = hit["text"].strip()
            context_lines.append(f"--- Excerpt {i} (Relevance: {1.0 / (1.0 + hit['distance']):.2f}) ---\n{text}")

        return "\n\n".join(context_lines)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns a list of all indexed document sources and their chunk count."""
        return self.store.list_sources(kind="document")

    def clear_documents(self) -> int:
        """Deletes all document chunks from the vector store."""
        return self.store.delete_by_kind("document")


def retrieve_document_context(query: str, store=None, embedder=None, k: int = 2) -> str:
    """Lightweight helper function for interaction and cognition loops."""
    if not store or not embedder or not query.strip():
        return ""
    try:
        rag = DocumentRAG(store=store, embedder=embedder)
        return rag.get_rag_context(query, k=k)
    except Exception as e:
        config.log_debug(f"[rag] retrieval error: {e}")
        return ""


# ------------------------------------------------------------------------------
# Standalone CLI Interface
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Karma Document RAG CLI (MarkItDown + sqlite-vec)")
    parser.add_argument("--ingest", "-i", type=str, help="Path to PDF or document file to ingest")
    parser.add_argument("--dir", "-d", type=str, help="Directory of PDF documents to ingest recursively")
    parser.add_argument("--query", "-q", type=str, help="Semantic search query against indexed documents")
    parser.add_argument("--list", "-l", action="store_true", help="List all indexed documents")
    parser.add_argument("--clear", action="store_true", help="Clear all indexed documents from vector store")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")

    args = parser.parse_args()
    rag = DocumentRAG()

    if args.clear:
        count = rag.clear_documents()
        print(f"✓ Cleared {count} document chunks from knowledge base.")
        return

    if args.list:
        sources = rag.list_documents()
        if not sources:
            print("No documents currently indexed.")
        else:
            print("\n📚 Indexed Knowledge Documents:")
            for s in sources:
                print(f"  • {s['source']}: {s['count']} chunks")
            print()
        return

    if args.ingest:
        rag.ingest_pdf(args.ingest, verbose=True)
        return

    if args.dir:
        rag.ingest_directory(args.dir)
        return

    if args.query:
        print(f"\n🔍 Query: '{args.query}'\n")
        ctx = rag.get_rag_context(args.query, k=args.k)
        if ctx:
            print(ctx)
        else:
            print("No relevant document knowledge found for this query.")
        print()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
