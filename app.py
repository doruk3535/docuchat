from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


st.set_page_config(
    page_title="CodeDocMate PDF Lite",
    page_icon="📄",
    layout="wide",
)

# --------------------- Styling ---------------------

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1200px;
    }
    .cd-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.3rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(56,189,248,0.12));
        border: 1px solid rgba(148,163,184,0.6);
        backdrop-filter: blur(10px);
    }
    .cd-pill {
        display:inline-flex;
        align-items:center;
        gap:0.35rem;
        padding:0.12rem 0.7rem;
        border-radius:999px;
        font-size:0.75rem;
        background:rgba(15,23,42,0.9);
        border:1px solid rgba(148,163,184,0.7);
        color:#e5e7eb;
    }
    .cd-section {
        padding: 1.0rem 1.2rem;
        border-radius: 1.0rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(30,64,175,0.9);
        box-shadow: 0 18px 36px rgba(15,23,42,0.6);
    }
    .cd-metric {
        padding: 0.6rem 0.8rem;
        border-radius: 0.8rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(148,163,184,0.55);
        font-size: 0.8rem;
    }
    .cd-metric h3 {
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
        color: #9ca3af;
    }
    .cd-metric span {
        font-size: 1.0rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.3rem;
        border: 1px solid rgba(56,189,248,0.7);
        background: linear-gradient(120deg, #0369a1, #0ea5e9);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stButton>button:hover {
        border-color: #e5e7eb;
        background: linear-gradient(120deg, #075985, #0284c7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------- Data structures ---------------------


@dataclass
class PdfChunk:
    id: int
    page_num: int
    text: str


@dataclass
class DocumentIndex:
    doc_name: str
    n_pages: int
    chunks: List[PdfChunk]
    vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray


# --------------------- PDF processing ---------------------


def extract_pdf_text(uploaded_file) -> List[str]:
    reader = PdfReader(uploaded_file)
    pages_text: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = txt.replace("\r", " ").replace("\u00A0", " ")
        pages_text.append(txt)
    return pages_text


def build_document_index(uploaded_pdf, name: str) -> DocumentIndex:
    pages_text = extract_pdf_text(uploaded_pdf)
    chunks: List[PdfChunk] = []
    texts: List[str] = []

    cid = 0
    for i, page_text in enumerate(pages_text, start=1):
        clean = page_text.strip()
        if not clean:
            continue
        chunks.append(PdfChunk(id=cid, page_num=i, text=clean))
        texts.append(clean)
        cid += 1

    if not texts:
        raise RuntimeError("No extractable text found in PDF.")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        max_features=20000,
    )
    tfidf_sparse = vectorizer.fit_transform(texts)
    tfidf_matrix = normalize(tfidf_sparse, norm="l2", axis=1).toarray()

    return DocumentIndex(
        doc_name=name,
        n_pages=len(pages_text),
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


# --------------------- Retrieval & summarization ---------------------


def retrieve_top_k(
    index: DocumentIndex,
    query: str,
    k: int = 3,
) -> List[tuple[PdfChunk, float]]:
    q_vec = index.vectorizer.transform([query])
    q_vec = normalize(q_vec, norm="l2", axis=1).toarray()[0]

    sims = np.dot(index.tfidf_matrix, q_vec)
    top_idx = np.argsort(sims)[::-1][:k]

    results: List[tuple[PdfChunk, float]] = []
    for idx in top_idx:
        results.append((index.chunks[idx], float(sims[idx])))
    return results


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def summarize_from_chunks(
    chunks_with_scores: List[tuple[PdfChunk, float]],
    query: str,
    max_sentences: int = 5,
) -> str:
    if not chunks_with_scores:
        return ""

    all_sentences: List[Tuple[int, str, float, int]] = []
    for rank, (chunk, score) in enumerate(chunks_with_scores):
        s_list = split_sentences(chunk.text)
        for idx_s, s in enumerate(s_list):
            all_sentences.append((len(all_sentences), s, score, rank))

    if not all_sentences:
        return ""

    q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 3]

    scored_sentences: List[tuple[float, int, str]] = []
    for global_pos, s, chunk_score, rank in all_sentences:
        s_low = s.lower()
        overlap = sum(s_low.count(tok) for tok in q_tokens) if q_tokens else 0
        pos_weight = 1.0 / (1.0 + rank)
        score = overlap + 0.3 * pos_weight + 0.2 * chunk_score
        scored_sentences.append((score, global_pos, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top = scored_sentences[: max_sentences * 3]

    top_sorted = sorted(top, key=lambda x: x[1])
    selected: List[str] = []
    used = set()
    for _, _, s in top_sorted:
        if s in used:
            continue
        selected.append(s)
        used.add(s)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected)


def build_answer(chunks_with_scores: List[tuple[PdfChunk, float]], query: str) -> str:
    lines: List[str] = []
    lines.append("### Answer")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")

    if not chunks_with_scores:
        lines.append("No relevant text found in the document.")
        return "\n".join(lines)

    summary = summarize_from_chunks(chunks_with_scores, query, max_sentences=4)
    if not summary:
        summary = "A concise summary could not be generated from the current text."

    lines.append("Summary:")
    lines.append(summary)
    lines.append("")

    lines.append("Relevant pages:")
    for chunk, score in chunks_with_scores:
        score_pct = round(score * 100, 1)
        preview = summarize_from_chunks([(chunk, score)], query, max_sentences=2)
        lines.append(
            f"- Page {chunk.page_num} · relevance ~ {score_pct}%\n"
            f"  {preview}"
        )

    return "\n".join(lines)


# --------------------- Session state ---------------------

if "doc_index" not in st.session_state:
    st.session_state.doc_index = None  # type: ignore

idx: Optional[DocumentIndex] = st.session_state.doc_index

n_pages = idx.n_pages if idx else 0
n_chunks = len(idx.chunks) if idx else 0
doc_name = idx.doc_name if idx else "—"

# --------------------- Hero + metrics ---------------------

st.markdown(
    f"""
    <div class="cd-hero">
      <div class="cd-pill">
        <span>📄 CodeDocMate PDF Lite</span>
        <span>local · no API</span>
      </div>
      <h1 style="margin-top:0.6rem; margin-bottom:0.3rem; font-size:1.9rem;">
        PDF understanding and summarization interface
      </h1>
      <p style="margin:0; font-size:0.95rem; color:#e5e7eb;">
        Upload a PDF, index its text and ask questions to obtain short, focused summaries
        of the most relevant parts of the document.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"""
        <div class="cd-metric">
          <h3>Pages indexed</h3>
          <span>{n_pages}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="cd-metric">
          <h3>Text chunks</h3>
          <span>{n_chunks}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="cd-metric">
          <h3>Active document</h3>
          <span>{doc_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# --------------------- Tabs ---------------------

tab_overview, tab_qna, tab_explorer = st.tabs(
    ["Overview", "PDF & Q&A", "Explorer"]
)

# Overview tab
with tab_overview:
    st.markdown('<div class="cd-section">', unsafe_allow_html=True)
    st.subheader("Overview", anchor=False)
    st.markdown(
        """
        This application builds a local TF-IDF index of the PDF text and
        answers questions by selecting and combining the most relevant sentences.
        No external APIs or language models are used.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# PDF & Q&A tab
with tab_qna:
    st.markdown('<div class="cd-section">', unsafe_allow_html=True)
    st.subheader("PDF & question", anchor=False)

    left, right = st.columns([1.05, 1.25])

    with left:
        uploaded_pdf = st.file_uploader("PDF", type=["pdf"])
        default_name = uploaded_pdf.name if uploaded_pdf is not None else "document.pdf"
        doc_name_input = st.text_input("Name", value=default_name)

        if uploaded_pdf is not None and doc_name_input:
            if st.button("Index PDF"):
                try:
                    index = build_document_index(uploaded_pdf, doc_name_input)
                    st.session_state.doc_index = index
                    st.success(
                        f"Indexed {index.n_pages} pages and {len(index.chunks)} text chunks."
                    )
                except Exception as e:
                    st.error(f"Indexing error: {e}")

    with right:
        if st.session_state.doc_index is None:
            st.info("No document indexed.")
        else:
            index = st.session_state.doc_index  # type: ignore
            query = st.text_area("Question")
            k = st.slider("Number of pages to consider", 1, 5, 3)

            if st.button("Get answer"):
                results = retrieve_top_k(index, query, k=k)
                answer_md = build_answer(results, query)
                st.markdown(answer_md)

                with st.expander("Relevant raw text", expanded=False):
                    for i, (chunk, score) in enumerate(results, start=1):
                        st.markdown(f"#### Page {chunk.page_num} · score={score:.3f}")
                        st.text(chunk.text)

    st.markdown("</div>", unsafe_allow_html=True)

# Explorer tab
with tab_explorer:
    st.markdown('<div class="cd-section">', unsafe_allow_html=True)
    st.subheader("Explorer", anchor=False)

    if st.session_state.doc_index is None:
        st.info("No document indexed.")
    else:
        index = st.session_state.doc_index  # type: ignore
        page_nums = sorted({ch.page_num for ch in index.chunks})
        selected_page = st.selectbox("Page", page_nums)

        if selected_page is not None:
            for ch in index.chunks:
                if ch.page_num == selected_page:
                    st.text(ch.text)
                    break

    st.markdown("</div>", unsafe_allow_html=True)



