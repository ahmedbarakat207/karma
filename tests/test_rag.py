import os
import tempfile
import pytest

from src.memory.store import MemoryStore
from src.memory.rag import DocumentRAG, retrieve_document_context


@pytest.fixture
def temp_store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = MemoryStore(db_path=db_path)
    yield store
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_pdf():
    pdf_bytes = (
        b'%PDF-1.4\n'
        b'1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n'
        b'2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n'
        b'3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n'
        b'4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n'
        b'5 0 obj<</Length 210>>stream\n'
        b'BT /F1 14 Tf 50 720 Td (# Technical Specifications) Tj\n'
        b'/F1 12 Tf 50 680 Td (Karma robot is powered by an AGM 12V 9Ah battery.) Tj\n'
        b'50 650 Td (The brain is a Raspberry Pi 4 B with 8GB RAM running Qwen 2.5.) Tj\n'
        b'ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n0000000212 00000 n \n0000000277 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n537\n%%EOF\n'
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = f.name
    yield pdf_path
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def test_markitdown_pdf_conversion(sample_pdf):
    rag = DocumentRAG()
    md_text = rag.parse_pdf(sample_pdf)
    assert len(md_text) > 0
    assert "12V 9Ah" in md_text or "Karma" in md_text or "Technical Specifications" in md_text


def test_semantic_markdown_chunking():
    rag = DocumentRAG(chunk_size=100)
    markdown = (
        "# Heading 1\n\n"
        "This is paragraph one about robotics.\n\n"
        "## Subsection A\n\n"
        "This is paragraph two describing the motor drivers and power bus.\n\n"
        "This is paragraph three describing the vision camera."
    )
    chunks = rag.chunk_markdown(markdown, "manual.pdf")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "[Document: manual.pdf" in chunk


def test_rag_ingest_and_query(temp_store, sample_pdf):
    rag = DocumentRAG(store=temp_store)
    count = rag.ingest_pdf(sample_pdf, verbose=False)
    assert count >= 1

    docs = rag.list_documents()
    assert len(docs) == 1
    assert docs[0]["source"] == os.path.basename(sample_pdf)
    assert docs[0]["count"] == count

    context = rag.get_rag_context("What battery does the robot use?")
    assert len(context) > 0
    assert os.path.basename(sample_pdf) in context


def test_rag_idempotent_reindexing(temp_store, sample_pdf):
    rag = DocumentRAG(store=temp_store)
    rag.ingest_pdf(sample_pdf, verbose=False)
    initial_count = temp_store.count()

    rag.ingest_pdf(sample_pdf, verbose=False)
    second_count = temp_store.count()

    assert initial_count == second_count


def test_rag_clear_documents(temp_store, sample_pdf):
    rag = DocumentRAG(store=temp_store)
    rag.ingest_pdf(sample_pdf, verbose=False)
    assert len(rag.list_documents()) == 1

    cleared = rag.clear_documents()
    assert cleared >= 1
    assert len(rag.list_documents()) == 0


def test_retrieve_document_context_helper(temp_store, sample_pdf):
    rag = DocumentRAG(store=temp_store)
    rag.ingest_pdf(sample_pdf, verbose=False)

    ctx = retrieve_document_context("battery voltage", store=temp_store, embedder=rag.embedder)
    assert len(ctx) > 0
