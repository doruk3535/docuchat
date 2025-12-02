import io
from typing import List, Tuple

import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ------------------------------------------------------
# DocuChat – Free RAG-Style PDF QA + Summarization
# (No OpenAI, English UI but works with Turkish/English PDFs)
#
# Layers:
# - User Interface Layer: Streamlit web app
# - Document Processing Layer: PDF parsing, cleaning, chunking
# - Processing & Retrieval Layer: embeddings + FAISS vector search
# - Answering Layer: semantic ranking + lightweight "rewrite" for QA
# - Summarization Layer: sentence embeddings + KMeans clustering
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
) -> Tuple[List[str], faiss.IndexFlatIP, int, int, str]:
    """
    End-to-end document processing:
    - parse PDF
    - clean text
    - chunk
    - embed chunks
    - build FAISS index

    Returns:
      chunks, index, num_pages, num_words, cleaned_text
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

    return chunks, index, num_pages, num_words, cleaned


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


# ------------------------ SENTENCE SPLITTING ------------------------
def split_sentences(text: str) -> List[str]:
    """
    Slightly better sentence splitter:
    - splits on '.', '!' and '?'
    - removes very short fragments
    Works fine for both Turkish and English.
    """
    text = text.replace("\n", " ")
    separators = [".", "!", "?"]
    for sep in separators:
        text = text.replace(sep, ".")
    parts = text.split(".")
    sentences = [s.strip() for s in parts if len(s.strip()) > 25]
    return sentences


# ------------------------ LIGHTWEIGHT REWRITE FOR QA ------------------------
def rewrite_answer(question: str, sentences: List[str]) -> str:
    """
    Create a short, human-like answer by:
    - merging extracted sentences
    - cleaning and de-duplicating entity names
    - special handling for 'who/kim/karakter' type questions
    Works for Turkish + English.
    """
    if not sentences:
        # Turkish fallback, çünkü sen genelde TR soruyorsun :)
        return "Belgede bu soruya doğrudan cevap veren bir bilgi bulunamadı."

    # 1) Merge all extracted sentences
    text = " ".join(sentences)

    # 2) Basic cleaning
    text = text.replace(" ,", ",")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()

    # 3) Extract candidate entities (proper-looking words)
    #    Büyük harfle başlayan ve 3+ harfli olanları al.
    words = text.replace(",", " ").split()
    candidates = [w.strip() for w in words if w and w[0].isupper() and len(w) > 2]
    unique_entities = sorted(set(candidates))

    lower_q = question.lower()

    # 4) If question is about characters/people
    if ("kim" in lower_q) or ("karakter" in lower_q) or ("who" in lower_q):
        if unique_entities:
            return "Bu belgede geçen kişiler: " + ", ".join(unique_entities) + "."

    # 5) Default short rewrite for other types of questions
    if len(text) > 280:
        text = text[:280].rsplit(" ", 1)[0] + "..."

    return text


# ------------------------ QA ANSWERING LAYER ------------------------
def generate_answer_from_context(
    embedder: SentenceTransformer,
    question: str,
    selected_chunks: List[str],
    top_n_sentences: int = 4,
) -> Tuple[str, List[str]]:
    """
    Improved extractive answering:
      - split selected chunks into sentences
      - embed each sentence and the question
      - compute cosine similarity
      - pick top-N sentences that are both:
          * highly relevant to the question
          * not near-duplicates of each other
      - then pass them through a lightweight "rewrite" step
        to sound more like an intelligent answer.
    """
    full_text = "\n ".join(selected_chunks)
    sentences = split_sentences(full_text)

    if not sentences:
        return "Belgede bu soruya doğrudan cevap veren bir bilgi bulunamadı.", []

    # Embed question and sentences
    q_emb = embed_texts(embedder, [question])      # (1, dim)
    sent_embs = embed_texts(embedder, sentences)   # (N, dim)

    sims = np.dot(q_emb, sent_embs.T)[0]           # (N,)

    # Sort sentences by similarity (desc)
    sorted_idx = np.argsort(-sims)

    # Threshold: if even the best match is low, say "not found"
    best_sim = sims[sorted_idx[0]]
    MIN_SIM_THRESHOLD = 0.30

    if best_sim < MIN_SIM_THRESHOLD:
        return (
            "I could not confidently answer this question from the document. "
            "The relevant information may not be present or is only loosely related.",
            [],
        )

    # Select top-N sentences with diversity (no near-duplicates)
    selected_sentences: List[str] = []
    selected_vectors: List[np.ndarray] = []

    MAX_SENTENCES = max(1, top_n_sentences)

    for idx in sorted_idx:
        if len(selected_sentences) >= MAX_SENTENCES:
            break

        candidate = sentences[idx].strip()
        cand_vec = sent_embs[idx]

        # Check similarity to already selected sentences (to avoid duplicates)
        is_duplicate = False
        for vec in selected_vectors:
            sim_to_selected = float(np.dot(cand_vec, vec))
            if sim_to_selected > 0.9:  # nearly identical sentence
                is_duplicate = True
                break

        if not is_duplicate:
            selected_sentences.append(candidate)
            selected_vectors.append(cand_vec)

    if not selected_sentences:
        # Fallback: at least the single best one
        best_idx = int(sorted_idx[0])
        selected_sentences = [sentences[best_idx].strip()]

    # 🔥 Burada "akıllı" rewrite devreye giriyor
    final_answer = rewrite_answer(question, selected_sentences)

    return final_answer, selected_sentences


# ------------------------ SUMMARIZATION LAYER ------------------------
def summarize_document(
    embedder: SentenceTransformer,
    cleaned_text: str,
    num_summary_sentences: int = 6,
) -> str:
    """
    Smart extractive summarization:
      - split whole document into sentences
      - embed all sentences
      - cluster with KMeans
      - pick one representative sentence per cluster
      - order them by original position and join as a coherent summary
    """
    sentences = split_sentences(cleaned_text)
    if not sentences:
        return "I could not extract any sentences from this document."

    if len(sentences) <= num_summary_sentences:
        return " ".join(sentences)

    # Embed all sentences
    sent_embs = embed_texts(embedder, sentences)

    k = min(num_summary_sentences, sent_embs.shape[0])
    if k <= 1:
        return sentences[0]

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(sent_embs)
    centers = kmeans.cluster_centers_

    selected_idx: List[int] = []

    for ci in range(k):
        cluster_indices = np.where(kmeans.labels_ == ci)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_vectors = sent_embs[cluster_indices]
        center = centers[ci]
        sims = np.dot(cluster_vectors, center)
        best_local = cluster_indices[int(np.argmax(sims))]
        selected_idx.append(best_local)

    # Sorted & unique indices
    selected_idx = sorted(set(selected_idx))
    selected_idx = selected_idx[:num_summary_sentences]

    summary_sentences = [sentences[i] for i in selected_idx]
    summary_text = " ".join(summary_sentences)

    return "In summary, " + summary_text


# ------------------------ STREAMLIT UI ------------------------
def main():
    st.set_page_config(page_title="DocuChat – Free RAG", layout="wide")

    embedder = load_embedder()

    # Sidebar – architecture + controls
    with st.sidebar:
        st.title("DocuChat – Architecture")

        st.markdown("### Mode")
        mode = st.radio(
            "Choose mode",
            ["Question Answering", "Document Summarization"],
            index=0,
        )

        st.markdown("### Layers")
        st.markdown("*User Interface Layer*")
        st.caption("Streamlit web app: file upload, question input, answer display.")

        st.markdown("*Document Processing Layer*")
        st.caption("PDF parsing, text cleaning, chunking.")

        st.markdown("*Processing & Retrieval Layer*")
        st.caption("Sentence embeddings + FAISS vector search.")

        st.markdown("*Answering Layer*")
        st.caption("Semantic sentence ranking + lightweight rewrite for QA.")

        st.markdown("*Summarization Layer*")
        st.caption("Sentence embeddings + KMeans clustering to build a global summary.")

        st.markdown("---")
        st.markdown("### Model & Index")
        st.write(f"Embedding model: {EMBEDDING_MODEL_NAME}")
        st.write("Vector index: FAISS (Inner Product, cosine on normalized vectors)")

        st.markdown("---")
        st.markdown("### Retrieval Settings (for QA)")

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

        st.markdown("---")
        st.markdown("### Summarization Settings")
        summary_sentences = st.slider(
            "Number of sentences in summary",
            min_value=3,
            max_value=10,
            step=1,
            value=6,
        )

    st.title("DocuChat – Chat with Your PDF (Free, No API Key)")

    st.markdown(
        """
        This prototype implements a *RAG-style pipeline without any paid APIs*.  
        You can use it in two modes:

        - *Question Answering:* Ask questions about the PDF and get rewritten, human-like answers.  
        - *Document Summarization:* Generate a smart, global summary of the whole PDF.  

        Under the hood it uses sentence embeddings, FAISS vector search and KMeans-based
        summarization, plus a lightweight rewrite step to behave more like an intelligent AI system.
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
        chunks, index, num_pages, num_words, cleaned_text = process_document(
            file_bytes=file_bytes,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    st.success("Document processed successfully.")
    st.write(f"- Pages: *{num_pages}*")
    st.write(f"- Approx. words: *{num_words}*")
    st.write(f"- Number of chunks: *{len(chunks)}*")

    st.markdown("---")

    # ----------------- MODE 1: QUESTION ANSWERING -----------------
    if mode == "Question Answering":
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
                st.markdown(f"*Q:* {q}")
                st.markdown(a)
                st.markdown("---")

            # Last interaction details
            last_q, last_a, last_ctx, last_scores, last_top_sents = st.session_state.history[-1]

            with st.expander("Details: retrieved chunks and similarity scores"):
                st.markdown("*FAISS similarity scores for the last question (higher = more relevant):*")
                st.write(last_scores)
                st.markdown("---")
                st.markdown("*Combined retrieved context (all selected chunks):*")
                st.write(last_ctx)

            with st.expander("Details: top-ranked sentences fed into the rewrite step"):
                for s in last_top_sents:
                    st.markdown(f"- {s}")

    # ----------------- MODE 2: DOCUMENT SUMMARIZATION -----------------
    else:
        st.subheader("Document Summarization")

        if st.button("Generate summary"):
            with st.spinner("Generating smart summary of the document..."):
                summary = summarize_document(
                    embedder,
                    cleaned_text,
                    num_summary_sentences=summary_sentences,
                )

            st.markdown("### Summary")
            st.write(summary)


if _name_ == "_main_":
    main()

