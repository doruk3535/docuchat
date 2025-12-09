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
    page_title="PDF QA",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #111827 0, #020617 50%, #020617 100%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1200px;
    }
    .app-hero {
        padding: 1.2rem 1.4rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(56,189,248,0.12));
        border: 1px solid rgba(148,163,184,0.6);
        backdrop-filter: blur(10px);
    }
    .app-pill {
        display:inline-flex;
        align-items:center;
        gap:0.35rem;
        padding:0.1rem 0.7rem;
        border-radius:999px;
        font-size:0.75rem;
        background:rgba(15,23,42,0.9);
        border:1px solid rgba(148,163,184,0.7);
        color:#e5e7eb;
    }
    .app-section {
        padding: 1.0rem 1.2rem;
        border-radius: 1.0rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(30,64,175,0.9);
        box-shadow: 0 16px 32px rgba(15,23,42,0.6);
    }
    .app-metric {
        padding: 0.6rem 0.8rem;
        border-radius: 0.8rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(148,163,184,0.55);
        font-size: 0.8rem;
    }
    .app-metric h3 {
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
        color: #9ca3af;
    }
    .app-metric span {
        font-size: 1.0rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
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
        raise RuntimeError("No text extracted from PDF.")

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


def summarize_chunk(text: str, query: str, max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 3]
    if not q_tokens:
        return " ".join(sentences[:max_sentences])

    scored: List[tuple[int, str]] = []
    for s in sentences:
        s_low = s.lower()
        score = sum(s_low.count(tok) for tok in q_tokens)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for sc, s in scored[:max_sentences] if sc > 0]

    if not best:
        best = sentences[:max_sentences]

    return " ".join(best)


def build_explanation(chunks_with_scores: List[tuple[PdfChunk, float]], query: str) -> str:
    lines = [
        "### Answer\n",
        f"Question: {query}\n",
        "",
    ]

    if not chunks_with_scores:
        lines.append("No relevant text found.")
        return "\n".join(lines)

    top_chunk, _ = chunks_with_scores[0]
    summary = summarize_chunk(top_chunk.text, query, max_sentences=3)

    lines.append("Summary:")
    lines.append(summary or "(no summary available)")
    lines.append("")

    lines.append("Sources:")
    for ch, score in chunks_with_scores:
        score_pct = round(score * 100, 1)
        preview = summarize_chunk(ch.text, query, max_sentences=2)
        lines.append(
            f"- Page {ch.page_num} · relevance ~ {score_pct}%\n"
            f"  {preview}\n"
        )

    return "\n".join(lines)


if "doc_index" not in st.session_state:
    st.session_state.doc_index = None  # type: ignore

idx: Optional[DocumentIndex] = st.session_state.doc_index

n_pages = idx.n_pages if idx else 0
n_chunks = len(idx.chunks) if idx else 0
doc_name = idx.doc_name if idx else "—"

st.markdown(
    f"""
    <div class="app-hero">
      <div class="app-pill">
        <span>📄 PDF QA</span>
        <span>local · no API</span>
      </div>
      <h1 style="margin-top:0.6rem; margin-bottom:0.3rem; font-size:1.8rem;">
        PDF question–answer interface
      </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div class="app-metric">
          <h3>Pages</h3>
          <span>{n_pages}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="app-metric">
          <h3>Text chunks</h3>
          <span>{n_chunks}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="app-metric">
          <h3>Document</h3>
          <span>{doc_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

tab_overview, tab_qna, tab_explorer = st.tabs(
    ["Overview", "Q&A", "Pages"]
)

with tab_overview:
    st.markdown('<div class="app-section">', unsafe_allow_html=True)
    st.subheader("Overview", anchor=False)
    st.markdown(
        """
        This interface indexes PDF text locally and answers questions
        by selecting relevant sentences from the document.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_qna:
    st.markdown('<div class="app-section">', unsafe_allow_html=True)
    st.subheader("PDF and question", anchor=False)

    col_left, col_right = st.columns([1.0, 1.2])

    with col_left:
        uploaded_pdf = st.file_uploader("PDF", type=["pdf"])
        name = uploaded_pdf.name if uploaded_pdf is not None else "document.pdf"

        doc_name_input = st.text_input("Name", value=name)

        if uploaded_pdf is not None and doc_name_input:
            if st.button("Index PDF"):
                try:
                    index = build_document_index(uploaded_pdf, doc_name_input)
                    st.session_state.doc_index = index
                    st.success(
                        f"Indexed {index.n_pages} pages."
                    )
                except Exception as e:
                    st.error(f"Indexing error: {e}")

    with col_right:
        if st.session_state.doc_index is None:
            st.info("No document indexed.")
        else:
            index = st.session_state.doc_index  # type: ignore
            query = st.text_area("Question")
            k = st.slider("Pages to use", 1, 5, 3)

            if st.button("Answer"):
                results = retrieve_top_k(index, query, k=k)
                explanation_md = build_explanation(results, query)
                st.markdown(explanation_md)

                with st.expander("Selected pages (raw text)", expanded=False):
                    for i, (ch, score) in enumerate(results, start=1):
                        st.markdown(f"#### Page {ch.page_num} · score={score:.3f}")
                        st.text(ch.text)

    st.markdown("</div>", unsafe_allow_html=True)

with tab_explorer:
    st.markdown('<div class="app-section">', unsafe_allow_html=True)
    st.subheader("Pages", anchor=False)

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


