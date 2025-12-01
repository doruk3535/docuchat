import io
from typing import List, Tuple

import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------
# DocuChat – Free RAG-Style PDF QA (No OpenAI, English UI)
#
# Architecture alignment:
# - User Interface Layer: Streamlit web app
# - Document Processing Layer: PDF parsing, cleaning, chunking
# - Processing & Retrieval Layer: embeddings + FAISS vector search
# - Answering Layer: sentence-level semantic retrieval (extractive)
# ------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ------------- MODEL LOADING (shared across sessions) -------------
@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load the sentence-transformers model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ------------------------ DOCUMENT PROCESSING ------------------------
def parse_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """
    Extract raw text and number of pages from a PDF.
    """
    pdf_io = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(pdf_io)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    full_text = "\n".join(pages)
    num_pages = len(reader.pages)
    return full_text, num_pages


def clean_text(text: str) -> str:
    """
    Simple cleaning: strip whitespace, remove empty lines, normalize spaces.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = " ".join(lines)
    cleaned = " ".join(cleaned.split())
    return cleaned


def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
    """
    Split the document into overlapping chunks.
    Here "tokens" ~ words (approximation is fine for our use case).
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


# ------------------------ EMBEDDINGS + FAISS ------------------------
def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Compute sentence-transformer embeddings and L2-normalize them
    so cosine similarity can be approximated by inner product.
    """
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norms
    return emb.astype("float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using inner product (cosine similarity on normalized vectors).
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


@st.cache_data(show_spinner=False)
def process_document(
    file_bytes: bytes,
    chunk_size: int,
    overlap: int,
    max_chunks: int = 100,
) -> Tuple[List[str], faiss.IndexFlatIP, int, int]:
    """
    End-to-end document processing:
    - parse PDF
    - clean text
    - chunk
    - embed chunks
    - build FAISS index

    Returns:
      chunks, index, num_pages, num_words
    """
    embedder = load_embedder()

    raw_text, num_pages = parse_pdf(file_bytes)
    cleaned = clean_text(raw_text)
    num_words = len(cleaned.split())

    chunks = chunk_text(cleaned, max_tokens=chunk_size, overlap=overlap)

    # Limit number of chunks for performance on very large PDFs
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    vectors = embed_texts(embedder, chunks)
    index = build_faiss_index(vectors)

    return chunks, index, num_pages, num_words


def retrieve_context(
    embedder: SentenceTransformer,
    question: str,
    chunks: List[str],
    index: faiss.IndexFlatIP,
    k: int = 4,
) -> Tuple[str, np.ndarray, List[str]]:
    """
    Given a user question:
      - embed the question
      - retrieve top-k most similar chunks from FAISS
      - return combined context, scores, and individual selected chunks
    """
    q_emb = embed_texts(embedder, [question])  # (1, dim)
    scores, indices = index.search(q_emb, k)
    selected_chunks = [chunks[i] for i in indices[0]]
    context_text = "\n\n---\n\n".join(selected_chunks)
    return context_text, scores[0], selected_chunks


# ------------------------ ANSWERING LAYER ------------------------
def split_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter based on periods.
    Good enough for a prototype + avoids external dependencies.
    """
    raw = text.replace("\n", " ")
    parts = raw.split(".")
    sentences = [s.strip() for s in parts if len(s.strip()) > 20]
    return sentences


def generate_answer_from_context(
    embedder: SentenceTransformer,
    question: str,
    selected_chunks: List[str],
    top_n_sentences: int = 4,
) -> Tuple[str, List[str]]:
    """
    Extractive "answering":
      - split selected chunks into sentences
      - embed each sentence and the question
      - compute cosine similarity
      - pick top-N most relevant sentences as the answer

    This simulates an Answering Layer without any paid LLM.
    """
    full_text = "\n ".join(selected_chunks)
    sentences = split_sentences(full_text)

    if not sentences:
        return "I could not find any sentence directly related to your question.", []

    q_emb = embed_texts(embedder, [question])    # (1, dim)
    sent_embs = embed_texts(embedder, sentences) # (N, dim)

    sims = np.dot(q_emb, sent_embs.T)[0]  # (N,)

    top_idx = np.argsort(-sims)[:top_n_sentences]
    top_sentences = [sentences[i].strip() for i in top_idx]

    answer_text = (
        f"**Question:** {question}\n\n"
        f"**Most relevant sentences found in the document:**\n\n"
        + "\n\n".join(f"- {s}" for s in top_sentences)
    )

    return answer_text, top_sentences


# ------------------------ STREAMLIT UI ------------------------
def main():
    st.set_page_config(page_title="DocuChat – Free RAG", layout="wide")

    embedder = load_embedder()

    # Sidebar – architecture + controls
    with st.sidebar:
        st.title("DocuChat – Architecture")

        st.markdown("### Layers")
        st.markdown("**User Interface Layer**")
        st.caption("Streamlit web app: file upload, question input, answer display.")

        st.markdown("**Document Processing Layer**")
        st.caption("PDF parsing, text cleaning, chunking.")

        st.markdown("**Processing & Retrieval Layer**")
        st.caption("Sentence embeddings + FAISS vector search.")

        st.markdown("**Answering Layer**")
        st.caption("Semantic sentence ranking (extractive answer).")

        st.markdown("---")
        st.markdown("### Model & Index")
        st.write(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
        st.write("Vector index: FAISS (Inner Product, cosine on normalized vectors)")

        st.markdown("---")
        st.markdown("### Retrieval Settings")

        chunk_size = st.slider(
            "Chunk size (words)",
            min_value=200,
            max_value=1200,
            step=100,
            value=800,
        )
        overlap = st.slider(
            "Chunk overlap (words)",
            min_value=0,
            max_value=400,
            step=50,
            value=100,
        )
        k_chunks = st.slider(
            "Top-k chunks to retrieve",
            min_value=1,
            max_value=10,
            step=1,
            value=4,
        )
        top_n_sentences = st.slider(
            "Top-N sentences for the answer",
            min_value=1,
            max_value=10,
            step=1,
            value=4,
        )

    st.title("DocuChat – Chat with Your PDF (Free, No API Key)")

    st.markdown(
        """
        This prototype implements a **RAG-style pipeline without any paid APIs**.  
        The system:
        1. Extracts and cleans text from the uploaded PDF  
        2. Splits it into overlapping **chunks**  
        3. Creates **semantic embeddings** of each chunk  
        4. Uses **FAISS** to retrieve the most relevant chunks for your question  
        5. Ranks individual sentences inside those chunks to build an **extractive answer**  
        """
    )

    if "history" not in st.session_state:
        # question, answer, context, scores, top_sentences
        st.session_state.history: List[Tuple[str, str, str, np.ndarray, List[str]]] = []

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file is None:
        st.info("To get started, please upload a PDF file.")
        return

    # Read file bytes once
    file_bytes = uploaded_file.read()

    # Process document (cached by bytes + settings)
    with st.spinner("Processing document (parsing, chunking, embeddings, FAISS index)..."):
        chunks, index, num_pages, num_words = process_document(
            file_bytes=file_bytes,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    st.success(f"Document processed successfully.")
    st.write(f"- Pages: **{num_pages}**")
    st.write(f"- Approx. words: **{num_words}**")
    st.write(f"- Number of chunks: **{len(chunks)}**")

    st.markdown("---")

    question = st.text_input("Ask a question about this document:")

    if question:
        with st.spinner("Retrieving relevant chunks and building an answer..."):
            context, scores, selected_chunks = retrieve_context(
                embedder,
                question,
                chunks,
                index,
                k=k_chunks,
            )

            answer, top_sentences = generate_answer_from_context(
                embedder,
                question,
                selected_chunks,
                top_n_sentences=top_n_sentences,
            )

        st.session_state.history.append(
            (question, answer, context, scores, top_sentences)
        )

    # Show answer & history
    if st.session_state.history:
        st.subheader("Question–Answer History")

        for q, a, _, _, _ in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(a)
            st.markdown("---")

        # Last interaction details
        last_q, last_a, last_ctx, last_scores, last_top_sents = st.session_state.history[-1]

        with st.expander("Details: retrieved chunks and similarity scores"):
            st.markdown("**FAISS similarity scores for the last question (higher = more relevant):**")
            st.write(last_scores)
            st.markdown("---")
            st.markdown("**Combined retrieved context (all selected chunks):**")
            st.write(last_ctx)

        with st.expander("Details: top-ranked sentences used in the answer"):
            for s in last_top_sents:
                st.markdown(f"- {s}")


if __name__ == "__main__":
    main()

