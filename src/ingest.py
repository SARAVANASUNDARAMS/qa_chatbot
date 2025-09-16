# src/ingest.py
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Download punkt tokenizer if not already
nltk.download("punkt")


def extract_text_by_page(pdf_path):
    """
    Extract text from each page of the PDF.
    Returns a list of dicts: [{"page": int, "text": str}, ...]
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages_text.append({"page": i, "text": page_text.replace("\n", " ")})
    return pages_text


def chunk_text(text, page_num, chunk_size=100, overlap=20):
    """
    Break text into chunks of chunk_size sentences with overlap.
    Returns a list of dicts: [{"page": page_num, "text": chunk_text}, ...]
    """
    sentences = sent_tokenize(text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunk = " ".join(sentences[start:end])
        chunks.append({"page": page_num, "text": chunk})
        start += chunk_size - overlap
    return chunks


def build_faiss_index(pages_text, model_name="all-MiniLM-L6-v2"):
    """
    Build FAISS index from PDF pages.
    Returns:
        - index: FAISS index
        - embeddings: numpy array of embeddings
        - all_chunks: list of dicts {"page": int, "text": str}
        - embedder: SentenceTransformer model
    """
    embedder = SentenceTransformer(model_name)
    all_chunks = []

    # Chunk all pages
    for page in pages_text:
        page_num = page["page"]
        text = page["text"]
        page_chunks = chunk_text(text, page_num)
        all_chunks.extend(page_chunks)

    # Embed all chunks
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings, all_chunks, embedder
