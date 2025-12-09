from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ============================================================
# 0. PAGE CONFIG & GLOBAL STYLE
# ============================================================

st.set_page_config(
    page_title="CodeDocMate Lite",
    page_icon="🧠",
    layout="wide",
)

# --- Custom CSS: navy blue / glassy UI ---
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    .codemate-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 1.4rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(56,189,248,0.14));
        border: 1px solid rgba(148,163,184,0.5);
        backdrop-filter: blur(14px);
    }
    .codemate-pill {
        display:inline-flex;
        align-items:center;
        gap:0.35rem;
        padding:0.12rem 0.65rem;
        border-radius:999px;
        font-size:0.75rem;
        background:rgba(15,23,42,0.9);
        border:1px solid rgba(148,163,184,0.7);
        color:#e5e7eb;
    }
    .codemate-section {
        padding: 1.2rem 1.4rem;
        border-radius: 1.2rem;
        background: rgba(15,23,42,0.92);
        border: 1px solid rgba(30,64,175,0.9);
        box-shadow: 0 20px 50px rgba(15,23,42,0.6);
    }
    .codemate-section-soft {
        padding: 1.0rem 1.1rem;
        border-radius: 1.0rem;
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(30,64,175,0.5);
    }
    .codemate-metric {
        padding: 0.7rem 0.9rem;
        border-radius: 0.9rem;
        background: rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.55);
        font-size: 0.8rem;
    }
    .codemate-metric h3 {
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
        color: #9ca3af;
    }
    .codemate-metric span {
        font-size: 1.05rem;
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
    .codemate-badge-muted {
        display:inline-flex;
        align-items:center;
        gap:0.3rem;
        padding:0.12rem 0.55rem;
        border-radius:999px;
        border:1px solid rgba(148,163,184,0.5);
        font-size:0.7rem;
        color:#9ca3af;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }
    .css-1l269bu, .css-1qhmv8l {  /* hide default Streamlit toolbar shadows if any */
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 1. DATA STRUCTURES & CONSTANTS
# ============================================================

SUPPORTED_EXT = [".py"]  # istersen buraya .js, .java vs ekleyebilirsin


@dataclass
class Chunk:
    """Single code chunk with metadata and index in TF-IDF matrix."""
    id: int
    file_path: str
    start_line: int
    end_line: int
    text: str


@dataclass
class ProjectIndex:
    """In-memory index for a single project."""
    project_id: str
    root_dir: Path
    files_df: pd.DataFrame          # columns: path, n_lines, n_chunks
    chunks: List[Chunk]
    vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray        # shape: (n_chunks, vocab)


# ============================================================
# 2. HELPER FUNCTIONS – FILES & CHUNKING
# ============================================================

def save_and_extract_zip(uploaded_file, extract_root: Path) -> Path:
    """Save uploaded ZIP to disk and extract all contents."""
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_path = extract_root / "project.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    return extract_root


def find_code_files(root: Path) -> List[Path]:
    """Recursively find all supported code files within root."""
    return [p for p in root.rglob("*") if p.suffix in SUPPORTED_EXT]


def read_file(path: Path) -> str:
    """Read a text file safely using UTF-8 encoding."""
    return path.read_text(encoding="utf-8", errors="ignore")


def simple_chunk(code: str, max_lines: int = 45) -> List[Tuple[int, int, str]]:
    """
    Split code into simple line-based chunks.
    Returns list of tuples: (start_line, end_line, chunk_text)
    """
    lines = code.splitlines()
    chunks: List[Tuple[int, int, str]] = []
    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i + max_lines]
        start = i + 1
        end = i + len(chunk_lines)
        chunks.append((start, end, "\n".join(chunk_lines)))
    return chunks


def extract_function_name_from_query(query: str) -> Optional[str]:
    """
    Try to extract a function name from a natural language question.
    Example: "What does calculate_loss() do?" -> "calculate_loss"
    """
    match = re.search(r"([a-zA-Z_]\w*)\s*\(", query)
    if match:
        return match.group(1)
    return None


# ============================================================
# 3. INDEXING & RETRIEVAL
# ============================================================

def build_project_index(uploaded_zip, project_id: str) -> ProjectIndex:
    """
    End-to-end indexing:
      1. Extract zip
      2. Find *.py files
      3. Chunk each file
      4. Build TF-IDF matrix (chunk-level)
    """
    root_dir = save_and_extract_zip(uploaded_zip, Path("projects") / project_id)

    code_files = find_code_files(root_dir)
    rows = []
    chunks: List[Chunk] = []
    chunk_texts: List[str] = []

    chunk_counter = 0

    for file_path in code_files:
        code = read_file(file_path)
        file_chunks = simple_chunk(code, max_lines=45)

        for start, end, text in file_chunks:
            chunks.append(
                Chunk(
                    id=chunk_counter,
                    file_path=str(file_path.relative_to(root_dir)),
                    start_line=start,
                    end_line=end,
                    text=text,
                )
            )
            chunk_texts.append(text)
            chunk_counter += 1

        rows.append(
            {
                "path": str(file_path.relative_to(root_dir)),
                "n_lines": len(code.splitlines()),
                "n_chunks": len(file_chunks),
            }
        )

    files_df = pd.DataFrame(rows).sort_values("path")

    if not chunk_texts:
        raise RuntimeError("No chunks produced – is the project empty or unsupported?")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        max_features=30_000,
    )
    tfidf_sparse = vectorizer.fit_transform(chunk_texts)
    tfidf_matrix = normalize(tfidf_sparse, norm="l2", axis=1).toarray()

    return ProjectIndex(
        project_id=project_id,
        root_dir=root_dir,
        files_df=files_df,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )


def retrieve_top_k(
    index: ProjectIndex,
    query: str,
    func_name: Optional[str],
    k: int = 5,
) -> List[Tuple[Chunk, float]]:
    """
    TF-IDF-based retrieval:
      1. Build a query string (+ function name if available)
      2. Transform into TF-IDF vector
      3. Compute cosine similarity with all chunks
      4. Return top-k (Chunk, score)
    """
    query_text = query
    if func_name:
        # function is important – add multiple times to boost weight
        query_text += f" function {func_name} {func_name}"

    q_vec = index.vectorizer.transform([query_text])
    q_vec = normalize(q_vec, norm="l2", axis=1).toarray()[0]  # (vocab,)

    sims = np.dot(index.tfidf_matrix, q_vec)  # shape: (n_chunks,)

    top_idx = np.argsort(sims)[::-1][:k]
    results: List[Tuple[Chunk, float]] = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        results.append((index.chunks[idx], float(sims[idx])))

    if not results:
        # fallback: lowest-indexed chunks
        for idx in range(min(k, len(index.chunks))):
            results.append((index.chunks[idx], 0.0))

    return results


def local_explanation(chunks_with_scores: List[Tuple[Chunk, float]], query: str) -> str:
    """
    Build a rule-based / stub explanation for retrieved chunks.
    """
    lines = [
        "### 🔍 Prototype explanation (local only)\n",
        "You asked:\n",
        f"> {query}\n",
        "CodeDocMate Lite scanned your project and highlighted these chunks:\n",
    ]

    for ch, score in chunks_with_scores:
        score_pct = round(score * 100, 1)
        preview = "\n".join(ch.text.splitlines()[:12])
        lines.append(
            f"- **File:** `{ch.file_path}` &nbsp; · &nbsp; "
            f"Lines **{ch.start_line}–{ch.end_line}** &nbsp; · &nbsp; "
            f"Relevance ~ **{score_pct}%**\n\n"
            f"  ```python\n{preview}\n  ```\n"
        )

    lines.append(
        "\nThis Lite build does **not** call any LLM. "
        "It surfaces potentially relevant code regions using TF-IDF based retrieval. "
        "In a full RAG + LLM edition, these snippets would be sent to a language model "
        "for deep, natural-language explanations."
    )

    return "\n".join(lines)


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

def init_state():
    if "project_index" not in st.session_state:
        st.session_state.project_index = None
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Overview"


init_state()

with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom:0.7rem;">
          <div class="codemate-pill">🧠 CodeDocMate Lite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("**Navigation**")
    page = st.radio(
        "",
        [
            "Overview",
            "Upload & Index",
            "Search & Explain",
            "File Explorer",
            "Chunk Browser",
            "Analytics",
            "Settings & About",
        ],
        index=[
            "Overview",
            "Upload & Index",
            "Search & Explain",
            "File Explorer",
            "Chunk Browser",
            "Analytics",
            "Settings & About",
        ].index(st.session_state.active_page),
    )
    st.session_state.active_page = page

    st.markdown("---")

    if st.session_state.project_index is None:
        st.caption("No active project indexed yet.")
    else:
        pi: ProjectIndex = st.session_state.project_index
        st.caption(f"**Active project:** `{pi.project_id}`")
        st.caption(f"{len(pi.files_df)} files · {len(pi.chunks)} chunks")


# ============================================================
# 5. HERO HEADER (always visible)
# ============================================================

pi: Optional[ProjectIndex] = st.session_state.project_index

n_files = pi.files_df.shape[0] if pi is not None else 0
n_chunks = len(pi.chunks) if pi is not None else 0
project_name = pi.project_id if pi is not None else "—"

st.markdown(
    f"""
    <div class="codemate-hero">
      <div class="codemate-pill">
        <span>Local-only · No API key</span>
        <span>TF-IDF code retrieval</span>
      </div>
      <h1 style="margin-top:0.6rem; margin-bottom:0.3rem; font-size:2.0rem;">
        Navy-blue CodeBase Intelligence Playground
      </h1>
      <p style="margin:0; font-size:0.95rem; color:#e5e7eb;">
        Upload a zipped Python project, index it into semantic chunks, and explore which
        parts of the code are most related to your questions — all without any external API.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(
        f"""
        <div class="codemate-metric">
          <h3>Indexed Python files</h3>
          <span>{n_files}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""
        <div class="codemate-metric">
          <h3>Total code chunks</h3>
          <span>{n_chunks}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f"""
        <div class="codemate-metric">
          <h3>Active project</h3>
          <span>{project_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")  # spacing


# ============================================================
# 6. PAGE IMPLEMENTATIONS
# ============================================================

def page_overview():
    st.markdown(
        """
        ### What is CodeDocMate Lite?

        **CodeDocMate Lite** is a *local-only* playground for exploring Retrieval-Augmented
        Generation (RAG) ideas on top of codebases – without actually calling any LLM.

        **Pipeline in this build:**

        1. Upload a zipped project.
        2. We scan the archive for supported code files (currently `.py`).
        3. Each file is split into line-based chunks (~45 lines per chunk).
        4. A TF-IDF vector is built for every chunk.
        5. When you ask a question, we:
           - detect function names in the question (e.g. `calculate_loss()`),
           - build a query vector using TF-IDF,
           - compute cosine similarity with all chunks,
           - return the top-k most relevant ones.
        6. Finally, a rule-based explanation is rendered around those chunks.

        The goal: **demo the RAG flow** clearly, with a professional UI, even when
        no OpenAI / LLM access is available.
        """
    )


def page_upload_index():
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Upload & index project", anchor=False)

    uploaded_zip = st.file_uploader(
        "Upload your project (.zip)",
        type=["zip"],
        help="For now, only `.py` files inside the ZIP are used for indexing.",
    )

    default_project_id = "demo-project"
    if uploaded_zip is not None:
        default_project_id = uploaded_zip.name
        if default_project_id.lower().endswith(".zip"):
            default_project_id = default_project_id[:-4]

    project_id = st.text_input("Project ID", value=default_project_id)

    if uploaded_zip is not None and project_id:
        st.info(
            "ZIP received. When you click the button below, "
            "CodeDocMate Lite will extract the project, scan for `.py` files, "
            "chunk them and build a TF-IDF index."
        )

        if st.button("Process & Index Project", type="primary"):
            try:
                index = build_project_index(uploaded_zip, project_id)
                st.session_state.project_index = index

                st.success(
                    f"Indexed project `{project_id}` with "
                    f"{index.files_df.shape[0]} files and {len(index.chunks)} chunks."
                )

                st.dataframe(
                    index.files_df,
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as e:
                st.error(f"Failed to index project: {e}")
    else:
        st.caption("Tip: start with a small toy project to understand the flow.")

    st.markdown('</div>', unsafe_allow_html=True)


def page_search_explain():
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Search & explain", anchor=False)

    if st.session_state.project_index is None:
        st.warning("No project indexed yet. Go to **Upload & Index** first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    index: ProjectIndex = st.session_state.project_index

    col_q1, col_q2 = st.columns([1.9, 1.1])

    with col_q1:
        default_q = "What does the function calculate_loss() do?"
        query = st.text_area(
            "Your question",
            value=default_q,
            height=130,
            help="Ask about functions, model training, preprocessing, etc.",
        )

    with col_q2:
        st.markdown(
            """
            <div class="codemate-section-soft">
              <div class="codemate-badge-muted">
                <span>⚙️ Mode</span><span>Local TF-IDF</span>
              </div>
              <p style="margin:0.4rem 0 0.3rem; font-size:0.8rem; color:#e5e7eb;">
                This build does not call any LLM. It uses TF-IDF over code chunks
                and a rule-based explanation template.
              </p>
              <ul style="font-size:0.78rem; padding-left:1.1rem; margin-top:0.35rem;">
                <li>Mention functions like <code>calculate_loss()</code>.</li>
                <li>Ask about data flow or model behaviour.</li>
                <li>Compare responsibilities of two files.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    k = st.slider("Number of chunks to retrieve", min_value=3, max_value=10, value=5)

    if st.button("Retrieve & explain", type="primary"):
        func_name = extract_function_name_from_query(query)
        if func_name:
            st.info(f"Detected function name in question: `{func_name}`")
        else:
            st.info("No explicit function name detected in the question.")

        results = retrieve_top_k(index, query, func_name, k=k)
        explanation_md = local_explanation(results, query)
        st.markdown(explanation_md)

        with st.expander("Show retrieved chunks in full", expanded=False):
            for rank, (chunk, score) in enumerate(results, start=1):
                st.markdown(
                    f"#### #{rank} · `{chunk.file_path}` · "
                    f"lines {chunk.start_line}–{chunk.end_line} · score={score:.3f}"
                )
                st.code(chunk.text, language="python")

    st.markdown('</div>', unsafe_allow_html=True)


def page_file_explorer():
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("File explorer", anchor=False)

    if st.session_state.project_index is None:
        st.warning("No project indexed yet. Go to **Upload & Index** first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    index: ProjectIndex = st.session_state.project_index

    st.caption("Overview of all indexed Python files in the current project.")
    st.dataframe(
        index.files_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    file_filter = st.text_input(
        "Filter by file path substring",
        value="",
        placeholder="e.g. utils, model, trainer",
    )
    if file_filter:
        mask = index.files_df["path"].str.contains(file_filter, case=False, na=False)
        filtered = index.files_df[mask]
        st.write(f"Filtered results: {filtered.shape[0]} file(s)")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


def page_chunk_browser():
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Chunk browser", anchor=False)

    if st.session_state.project_index is None:
        st.warning("No project indexed yet. Go to **Upload & Index** first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    index: ProjectIndex = st.session_state.project_index

    file_paths = sorted({ch.file_path for ch in index.chunks})
    selected_file = st.selectbox("Select file", file_paths)

    file_chunks = [ch for ch in index.chunks if ch.file_path == selected_file]
    st.caption(f"{len(file_chunks)} chunk(s) in this file.")

    chunk_labels = [
        f"Lines {ch.start_line}–{ch.end_line}" for ch in file_chunks
    ]
    selected_label = st.selectbox("Select chunk", chunk_labels)

    selected_chunk = file_chunks[chunk_labels.index(selected_label)]
    st.code(selected_chunk.text, language="python")

    st.markdown('</div>', unsafe_allow_html=True)


def page_analytics():
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Analytics", anchor=False)

    if st.session_state.project_index is None:
        st.warning("No project indexed yet. Go to **Upload & Index** first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    index: ProjectIndex = st.session_state.project_index

    st.caption("Basic analytics for the indexed project.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top files by line count**")
        top_lines = index.files_df.sort_values("n_lines", ascending=False).head(10)
        st.bar_chart(
            data=top_lines.set_index("path")["n_lines"],
            use_container_width=True,
        )

    with col_b:
        st.markdown("**Top files by number of chunks**")
        top_chunks = index.files_df.sort_values("n_chunks", ascending=False).head(10)
        st.bar_chart(
            data=top_chunks.set_index("path")["n_chunks"],
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "You can extend this page with more advanced analytics: "
        "dependency graphs, function-level stats, test coverage overlays, etc."
    )

    st.markdown('</div>', unsafe_allow_html=True)


def page_settings():
    st.markdown('<div class="codemate-section">', unsafe_allow_html=True)
    st.subheader("Settings & About", anchor=False)

    st.markdown(
        """
        #### Configuration

        This Lite build is designed to be **safe** and **portable**:

        - No API keys required
        - No external network calls
        - All indexing happens in memory
        - TF-IDF based retrieval using `scikit-learn`

        You can extend it by:

        - Adding more file extensions to `SUPPORTED_EXT`
        - Plugging in a real vector store instead of in-memory TF-IDF
        - Adding an LLM call after retrieval to generate richer explanations
        """
    )

    st.markdown("---")

    st.markdown(
        """
        #### Architecture sketch

        - **UI**: Streamlit, multi-page style via sidebar navigation  
        - **Indexing**:
            - Uploaded ZIP → extracted to `projects/<project_id>`  
            - `find_code_files()` scans for supported files  
            - `simple_chunk()` breaks them into ~45-line chunks  
        - **Vectorization**:
            - `TfidfVectorizer` builds vectors for all chunks  
            - cosine similarity over normalized vectors  
        - **State**:
            - `ProjectIndex` dataclass stored in `st.session_state.project_index`  

        This structure is intentionally modular so you can:
        - Swap TF-IDF with sentence embeddings,
        - Replace local retrieval with FAISS / Elasticsearch,
        - or add a `RAG + LLM` mode next to the current local mode.
        """
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 7. ROUTING
# ============================================================

if page == "Overview":
    page_overview()
elif page == "Upload & Index":
    page_upload_index()
elif page == "Search & Explain":
    page_search_explain()
elif page == "File Explorer":
    page_file_explorer()
elif page == "Chunk Browser":
    page_chunk_browser()
elif page == "Analytics":
    page_analytics()
elif page == "Settings & About":
    page_settings()

