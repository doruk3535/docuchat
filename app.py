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
    page_title="Document QA",
    page_icon="📄",
    layout="wide",
)

# --------------------- basic styling (grey layout) ---------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e5e7eb;
    }
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1.8rem;
        max-width: 1250px;
    }
    .app-header {
        padding: 0.9rem 1.0rem;
        border-radius: 0.75rem;
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        margin-bottom: 0.8rem;
    }
    .panel {
        padding: 0.9rem 1.0rem;
        border-radius: 0.75rem;
        background-color: #f9fafb;
        border: 1px solid #d1d5db;
    }
    .metric-box {
        padding: 0.5rem 0.7rem;
        border-radius: 0.6rem;
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        font-size: 0.82rem;
    }
    .metric-box h3 {
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 0.12rem;
        color: #4b5563;
    }
    .metric-box span {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        border: 1px solid #9ca3af;
        background-color: #111827;
        color: #f9fafb;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stButton>button:hover {
        background-color: #374151;
        border-color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------- data containers ---------------------


@dataclass
class PdfBlock:
    id: int
    page_num: int
    text: str


@dataclass
class DocIndex:
    name: str
    n_pages: int
    blocks: List[PdfBlock]
    vectorizer: TfidfVectorizer
    matrix: np.ndarray


# --------------------- PDF parsing and indexing ---------------------


def _extract_text(pdf_file) -> List[str]:
    reader = PdfReader(pdf_file)
    out: List[str] = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        cleaned = raw.replace("\r", " ").replace("\u00A0", " ")
        out.append(cleaned)
    return out


def build_index(pdf_file, name: str) -> DocIndex:
    pages = _extract_text(pdf_file)
    blocks: List[PdfBlock] = []
    texts: List[str] = []
    bid = 0

    for i, page_text in enumerate(pages, start=1):
        t = page_text.strip()
        if not t:
            continue
        blocks.append(PdfBlock(id=bid, page_num=i, text=t))
        texts.append(t)
        bid += 1

    if not texts:
        raise RuntimeError("No textual content detected in the document.")

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        max_features=25000,
    )
    m_sparse = vec.fit_transform(texts)
    m = normalize(m_sparse, norm="l2", axis=1).toarray()

    return DocIndex(
        name=name,
        n_pages=len(pages),
        blocks=blocks,
        vectorizer=vec,
        matrix=m,
    )


# --------------------- retrieval ---------------------


def retrieve_blocks(index: DocIndex, query: str, k: int) -> List[tuple[PdfBlock, float]]:
    q_vec = index.vectorizer.transform([query])
    q_vec = normalize(q_vec, norm="l2", axis=1).toarray()[0]

    sims = np.dot(index.matrix, q_vec)
    order = np.argsort(sims)[::-1][:k]

    out: List[tuple[PdfBlock, float]] = []
    for idx in order:
        out.append((index.blocks[idx], float(sims[idx])))
    return out


# --------------------- sentence utilities ---------------------


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def make_summary(
    blocks_with_scores: List[tuple[PdfBlock, float]],
    query: str,
    max_sentences: int,
) -> str:
    if not blocks_with_scores:
        return ""

    all_sentences: List[Tuple[int, str, float, int]] = []
    global_pos = 0

    for rank, (block, score) in enumerate(blocks_with_scores):
        s_list = split_into_sentences(block.text)
        for s in s_list:
            all_sentences.append((global_pos, s, score, rank))
            global_pos += 1

    if not all_sentences:
        return ""

    q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 3]

    scored: List[tuple[float, int, str]] = []
    for pos, s, block_score, rank in all_sentences:
        s_low = s.lower()
        overlap = sum(s_low.count(tok) for tok in q_tokens) if q_tokens else 0
        rank_factor = 1.0 / (1.0 + rank)
        score = overlap + 0.25 * block_score + 0.2 * rank_factor
        scored.append((score, pos, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = scored[: max_sentences * 3]

    ordered = sorted(candidates, key=lambda x: x[1])
    selected: List[str] = []
    used = set()

    for _, _, s in ordered:
        if s in used:
            continue
        selected.append(s)
        used.add(s)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected)


def make_answer(
    blocks_with_scores: List[tuple[PdfBlock, float]],
    query: str,
    max_summary_sent: int,
    max_preview_sent: int,
) -> str:
    lines: List[str] = []
    lines.append("### Answer")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")

    if not blocks_with_scores:
        lines.append("No relevant text could be selected from the document.")
        return "\n".join(lines)

    summary = make_summary(blocks_with_scores, query, max_summary_sent)
    if not summary:
        summary = "A compact summary could not be created from the available text."

    lines.append("Summary:")
    lines.append(summary)
    lines.append("")

    lines.append("Relevant sections:")
    for block, score in blocks_with_scores:
        score_pct = round(score * 100, 1)
        preview = make_summary([(block, score)], query, max_preview_sent)
        lines.append(
            f"- Page {block.page_num} · relevance ~ {score_pct}%\n"
            f"  {preview}"
        )

    return "\n".join(lines)


# --------------------- session state ---------------------

if "doc_index" not in st.session_state:
    st.session_state.doc_index = None  # type: ignore

index: Optional[DocIndex] = st.session_state.doc_index

pages_count = index.n_pages if index else 0
blocks_count = len(index.blocks) if index else 0
doc_name = index.name if index else "—"

# --------------------- header ---------------------

st.markdown(
    f"""
    <div class="app-header">
      <h2 style="margin:0; font-size:1.25rem; color:#111827;">
        Document analysis workspace
      </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.markdown(
        f"""
        <div class="metric-box">
          <h3>Pages</h3>
          <span>{pages_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with mcol2:
    st.markdown(
        f"""
        <div class="metric-box">
          <h3>Text units</h3>
          <span>{blocks_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with mcol3:
    st.markdown(
        f"""
        <div class="metric-box">
          <h3>Document</h3>
          <span>{doc_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# --------------------- main layout: sidebar + two panels ---------------------

with st.sidebar:
    st.markdown("#### Controls")

    k_pages = st.slider(
        "Pages used for answering",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    max_summary_sent = st.slider(
        "Summary sentence limit",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
    )

    max_preview_sent = st.slider(
        "Per-page preview sentences",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )

    st.markdown("---")
    if index:
        st.caption(
            f"Indexed: {index.n_pages} pages, {len(index.blocks)} text units."
        )
    else:
        st.caption("No document indexed yet.")

left_col, right_col = st.columns([1.0, 1.3])

# left: pdf upload + info
with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Document")

    pdf_file = st.file_uploader("PDF file", type=["pdf"])
    default_name = pdf_file.name if pdf_file is not None else "document.pdf"
    doc_input_name = st.text_input("Name", value=default_name)

    if pdf_file is not None and doc_input_name:
        if st.button("Build index"):
            try:
                new_index = build_index(pdf_file, doc_input_name)
                st.session_state.doc_index = new_index
                index = new_index
                st.success(
                    f"Indexed {new_index.n_pages} pages and {len(new_index.blocks)} text units."
                )
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# right: question + answer
with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Question and answer")

    if index is None:
        st.info("No document is currently indexed.")
    else:
        question = st.text_area("Question", value="", height=120)

        if st.button("Generate answer"):
            blocks_scored = retrieve_blocks(index, question, k=k_pages)
            answer_text = make_answer(
                blocks_scored,
                question,
                max_summary_sent=max_summary_sent,
                max_preview_sent=max_preview_sent,
            )
            st.markdown(answer_text)

            with st.expander("Selected raw text", expanded=False):
                for block, score in blocks_scored:
                    st.markdown(f"**Page {block.page_num} · score={score:.3f}**")
                    st.text(block.text)

    st.markdown("</div>", unsafe_allow_html=True)
