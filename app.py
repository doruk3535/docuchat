"""
DocuChat Enterprise Edition (Demo)
----------------------------------
Local, secure RAG system for PDF question-answering.
This is a demo version for course presentation.
"""

import io
import time
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import plotly.express as px

# ==============================================================================
# 1. SYSTEM CONFIGURATION & LOGGING SETUP
# ==============================================================================

st.set_page_config(
page_title="DocuChat Demo",
page_icon="📝",
layout="wide",
initial_sidebar_state="expanded",
menu_items={
"About": "# DocuChat Enterprise Demo\nLocal RAG System v3.5",
},
)


class SystemLogger:
"""
   Handles application-wide logging to display in the UI and console.
   """

def __init__(self) -> None:
self.logs: List[str] = []
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger("DocuChat")

def info(self, message: str) -> None:
timestamp = datetime.now().strftime("%H:%M:%S")
entry = f"[{timestamp}] [INFO] {message}"
self.logs.append(entry)
self.logger.info(message)

def error(self, message: str) -> None:
timestamp = datetime.now().strftime("%H:%M:%S")
entry = f"[{timestamp}] [ERROR] {message}"
self.logs.append(entry)
self.logger.error(message)

def warning(self, message: str) -> None:
timestamp = datetime.now().strftime("%H:%M:%S")
entry = f"[{timestamp}] [WARN] {message}"
self.logs.append(entry)
self.logger.warning(message)

def get_logs(self) -> str:
return "\n".join(self.logs)


# Initialize Logger in Session State
if "sys_logger" not in st.session_state:
st.session_state.sys_logger = SystemLogger()

logger: SystemLogger = st.session_state.sys_logger

# Custom CSS
st.markdown(
"""
<style>
   /* Main area */
   .main {
       background-color: #0f172a;
       font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       color: #e5e7eb;
   }

   h1, h2, h3 {
       color: #38bdf8;
   }

   /* Chat bubbles */
   .user-bubble {
       align-self: flex-end;
       background-color: #1d4ed8;
       border: 1px solid #60a5fa;
       color: #e5e7eb;
       padding: 15px;
       border-radius: 15px 15px 0 15px;
       margin-left: 20%;
       box-shadow: 0 2px 4px rgba(0,0,0,0.3);
   }
   .bot-bubble {
       align-self: flex-start;
       background-color: #020617;
       border: 1px solid #334155;
       color: #e5e7eb;
       padding: 20px;
       border-radius: 15px 15px 15px 0;
       margin-right: 10%;
       box-shadow: 0 4px 6px rgba(0,0,0,0.4);
   }

   .source-metadata {
       font-size: 0.8rem;
       color: #9ca3af;
       margin-bottom: 5px;
       font-weight: 600;
       text-transform: uppercase;
       letter-spacing: 0.05em;
   }
   .source-content {
       font-family: 'Georgia', serif;
       border-left: 4px solid #38bdf8;
       padding-left: 10px;
       margin-top: 5px;
       background-color: #020617;
   }

   div[data-testid="stMetricValue"] {
       font-size: 1.5rem;
       color: #38bdf8;
   }

   /* Sidebar dark mode */
   section[data-testid="stSidebar"] {
       background-color: #020617;
   }
   section[data-testid="stSidebar"] * {
       color: #e5e7eb !important;
   }
   section[data-testid="stSidebar"] .stSlider label {
       color: #e5e7eb !important;
   }
</style>
""",
unsafe_allow_html=True,
)

# ==============================================================================
# 2. DATA STRUCTURES
# ==============================================================================


@dataclass
class DocumentChunk:
id: str
text: str
source_file: str
page_number: int
char_count: int
word_count: int

def to_dict(self) -> Dict:
return asdict(self)


@dataclass
class SearchResult:
chunk: DocumentChunk
score: float
relevance_label: str


# ==============================================================================
# 3. TEXT PROCESSING ENGINE (ETL Layer)
# ==============================================================================


class TextProcessor:
@staticmethod
def clean_text(text: str) -> str:
if not text:
return ""

text = text.encode("utf-8", "ignore").decode("utf-8")
text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
text = text.replace("\n", " ")
text = re.sub(r"\s+", " ", text)

if len(text) < 5:
return ""

return text.strip()


class DocumentIngestor:
def __init__(self) -> None:
self.processor = TextProcessor()

def process_pdf(
self, file_obj, chunk_size: int, overlap: int
) -> List[DocumentChunk]:
chunks: List[DocumentChunk] = []
try:
pdf_bytes = file_obj.read()
pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
file_name = file_obj.name

logger.info(
f"Started processing file: {file_name} ({len(pdf_reader.pages)} pages)"
)

for page_idx, page in enumerate(pdf_reader.pages):
raw_text = page.extract_text()
clean_text = self.processor.clean_text(raw_text)

if not clean_text:
continue

words = clean_text.split()
if not words:
continue

step = max(chunk_size - overlap, 1)
for i in range(0, len(words), step):
chunk_words = words[i : i + chunk_size]
chunk_str = " ".join(chunk_words)

if len(chunk_words) > 10:
chunk_obj = DocumentChunk(
id=f"{file_name}p{page_idx+1}{i}",
text=chunk_str,
source_file=file_name,
page_number=page_idx + 1,
char_count=len(chunk_str),
word_count=len(chunk_words),
)
chunks.append(chunk_obj)

logger.info(
f"Finished processing {file_name}. Generated {len(chunks)} chunks."
)
return chunks

except Exception as e:
logger.error(f"Error reading PDF {file_obj.name}: {str(e)}")
st.error(f"Failed to process {file_obj.name}. See logs for details.")
return []


# ==============================================================================
# 4. VECTOR DATABASE & SEARCH ENGINE
# ==============================================================================


class VectorDatabase:
def __init__(self, model_name: str) -> None:
self.model_name = model_name
self.model: Optional[SentenceTransformer] = None
self.index: Optional[faiss.IndexFlatL2] = None
self.chunks_map: Dict[int, DocumentChunk] = {}
self.is_initialized: bool = False

def load_model(self) -> None:
if self.model is None:
logger.info(f"Loading embedding model: {self.model_name}")
with st.spinner(f"Initializing AI Core ({self.model_name})..."):
self.model = SentenceTransformer(self.model_name)

def build_index(self, chunks: List[DocumentChunk]) -> bool:
self.load_model()

if not chunks:
logger.warning("No chunks provided to build index.")
return False

logger.info(f"Generating embeddings for {len(chunks)} chunks...")
texts = [c.text for c in chunks]
embeddings = self.model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

self.index = index
self.chunks_map = {i: c for i, c in enumerate(chunks)}
self.is_initialized = True

logger.info("Vector index built successfully.")
return True

def query(self, query_text: str, top_k: int = 3) -> List[SearchResult]:
if not self.is_initialized:
logger.error("Attempted to query uninitialized database.")
return []

self.load_model()
query_vec = self.model.encode([query_text]).astype("float32")
distances, indices = self.index.search(query_vec, top_k)

results: List[SearchResult] = []

for i, idx in enumerate(indices[0]):
if idx == -1:
continue

score = float(distances[0][i])
chunk = self.chunks_map.get(int(idx))
if not chunk:
continue

if score < 0.90:
label = "HIGH"
elif score < 1.30:
label = "MEDIUM"
else:
label = "LOW"

results.append(SearchResult(chunk=chunk, score=score, relevance_label=label))

return results

def get_embeddings_for_viz(
self,
) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
if not self.is_initialized:
return None, [], []

self.load_model()
full_texts = [c.text for c in self.chunks_map.values()]
embeddings = self.model.encode(full_texts, show_progress_bar=False)
texts = [c.text[:50] + "..." for c in self.chunks_map.values()]
sources = [c.source_file for c in self.chunks_map.values()]
return np.array(embeddings), texts, sources


# ==============================================================================
# 5. SESSION STATE MANAGEMENT
# ==============================================================================


class SessionManager:
@staticmethod
def initialize() -> None:
if "db" not in st.session_state:
st.session_state.db = VectorDatabase(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)
if "ingestor" not in st.session_state:
st.session_state.ingestor = DocumentIngestor()
if "chat_history" not in st.session_state:
st.session_state.chat_history: List[Dict[str, Any]] = []
if "processed_files" not in st.session_state:
st.session_state.processed_files = set()
if "stats" not in st.session_state:
st.session_state.stats = {"total_words": 0, "total_chunks": 0}

@staticmethod
def add_message(role: str, content: str) -> None:
st.session_state.chat_history.append(
{
"role": role,
"content": content,
"timestamp": datetime.now().isoformat(),
}
)

@staticmethod
def clear_history() -> None:
st.session_state.chat_history = []
logger.info("Chat history cleared by user.")


# ==============================================================================
# 6. USER INTERFACE COMPONENTS
# ==============================================================================


def render_sidebar() -> Tuple[int, int, int, bool]:
with st.sidebar:
# Takvim görseli kaldırıldı, sade başlık
st.title("System Control")
st.caption(f"Demo v3.5.0 | {datetime.now().strftime('%Y-%m-%d')}")

st.divider()

st.subheader("1. Ingestion Settings")
chunk_size = st.slider(
"Chunk Size (Tokens)",
min_value=100,
max_value=1000,
value=500,
help="Number of words per chunk. Smaller = more specific answers.",
)
overlap = st.slider(
"Overlap Window",
min_value=0,
max_value=200,
value=50,
help="Words shared between chunks to preserve context.",
)

st.divider()

st.subheader("2. Search Parameters")
top_k = st.slider("Retrieval Depth (K)", 1, 10, 3)
strict_mode = st.checkbox(
"Strict Relevance Filter",
value=True,
help="Hides low confidence results.",
)

st.divider()

col1, col2 = st.columns(2)
with col1:
if st.button("Clear Chat", use_container_width=True):
SessionManager.clear_history()
st.rerun()
with col2:
if st.button("Reset All", use_container_width=True):
st.session_state.clear()
st.rerun()

st.divider()
with st.expander("System Logs", expanded=False):
st.text_area("", value=logger.get_logs(), height=200, disabled=True)

return chunk_size, overlap, top_k, strict_mode


def render_analytics_tab(db: VectorDatabase) -> None:
st.header("📊 Knowledge Base Analytics")

if not db.is_initialized:
st.info("No data available. Please upload and process documents first.")
return

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Chunks", st.session_state.stats["total_chunks"])
col2.metric("Total Words", f"{st.session_state.stats['total_words']:,}")
col3.metric("Processed Files", len(st.session_state.processed_files))
col4.metric("Embedding Dim", "384")

st.markdown("---")

st.subheader("Semantic Vector Space (PCA Reduced)")
with st.spinner("Calculating 3D Projection..."):
embeddings, labels, sources = db.get_embeddings_for_viz()

if embeddings is not None and len(embeddings) > 2:
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
projections = pca.fit_transform(embeddings)

df_viz = pd.DataFrame(projections, columns=["x", "y", "z"])
df_viz["text"] = labels
df_viz["source"] = sources

fig = px.scatter_3d(
df_viz,
x="x",
y="y",
z="z",
color="source",
hover_data=["text"],
title="3D Document Cluster Map",
template="plotly_dark",
height=600,
)
st.plotly_chart(fig, use_container_width=True)
else:
st.warning("Not enough data points for 3D visualization.")

st.subheader("Chunk Length Distribution")
chunk_lengths = [c.word_count for c in db.chunks_map.values()]
fig_hist = px.histogram(
x=chunk_lengths,
nbins=20,
labels={"x": "Word Count"},
title="Distribution of Information Density",
template="plotly_dark",
)
st.plotly_chart(fig_hist, use_container_width=True)


def render_file_upload_area(chunk_size: int, overlap: int) -> None:
st.subheader("📂 Document Ingestion")

uploaded_files = st.file_uploader(
"Upload PDF Files (Multi-file support)", type=["pdf"], accept_multiple_files=True
)

if not uploaded_files:
return

new_files = [
f for f in uploaded_files if f.name not in st.session_state.processed_files
]

if new_files:
if st.button(f"Process {len(new_files)} New Files", type="primary"):
all_chunks: List[DocumentChunk] = []
progress_bar = st.progress(0, text="Starting ingestion...")

for i, file in enumerate(new_files):
chunks = st.session_state.ingestor.process_pdf(file, chunk_size, overlap)
all_chunks.extend(chunks)
st.session_state.processed_files.add(file.name)
progress_bar.progress(
(i + 1) / len(new_files), text=f"Processed {file.name}"
)

if "all_chunks" in st.session_state:
st.session_state.all_chunks.extend(all_chunks)
else:
st.session_state.all_chunks = all_chunks

if st.session_state.all_chunks:
success = st.session_state.db.build_index(st.session_state.all_chunks)
if success:
st.session_state.stats["total_chunks"] = len(
st.session_state.all_chunks
)
st.session_state.stats["total_words"] = sum(
c.word_count for c in st.session_state.all_chunks
)
st.success(
f"Successfully indexed {len(st.session_state.all_chunks)} chunks!"
)
time.sleep(1)
st.rerun()

progress_bar.empty()
else:
st.success("All uploaded files are processed and ready.")


def render_chat_interface(top_k: int, strict_mode: bool) -> None:
st.subheader("💬 AI Research Assistant")

if not st.session_state.db.is_initialized:
st.warning("⚠ System Offline. Please process documents in the 'Upload' tab.")
return

for msg in st.session_state.chat_history:
if msg["role"] == "user":
st.markdown(
f"<div class='user-bubble'>👤 <b>You:</b><br>{msg['content']}</div>",
unsafe_allow_html=True,
)
else:
st.markdown(
f"<div class='bot-bubble'>🤖 <b>System:</b><br>{msg['content']}</div>",
unsafe_allow_html=True,
)

prompt = st.chat_input("Ask a specific question about your documents...")

if prompt:
SessionManager.add_message("user", prompt)
st.rerun()

if (
st.session_state.chat_history
and st.session_state.chat_history[-1]["role"] == "user"
):
last_query = st.session_state.chat_history[-1]["content"]

with st.spinner("Analyzing semantic vectors..."):
time.sleep(0.3)
results = st.session_state.db.query(last_query, top_k=top_k)

response_html = ""
found_valid_result = False

if not results:
response_html = (
"<i>No relevant information found in the provided documents.</i>"
)
else:
response_html += (
f"<div style='margin-bottom:10px;'><b>Found "
f"{len(results)} references:</b></div>"
)

for res in results:
if strict_mode and res.relevance_label == "LOW":
continue

found_valid_result = True

if res.relevance_label == "HIGH":
color = "#22c55e"
elif res.relevance_label == "MEDIUM":
color = "#eab308"
else:
color = "#f97316"

response_html += f"""
                   <div style="border-left: 5px solid {color}; padding-left: 15px; margin-bottom: 20px; background: #020617; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.5);">
                       <div class="source-metadata">
                           📄 {res.chunk.source_file} | Page {res.chunk.page_number} | 
                           <span style="color:{color}; font-weight:bold;">{res.relevance_label} CONFIDENCE ({res.score:.3f})</span>
                       </div>
                       <div class="source-content">
                           "{res.chunk.text}"
                       </div>
                   </div>
                   """

if strict_mode and not found_valid_result:
response_html = (
"<i>Matching results were found but filtered out due to low "
"relevance (Strict Mode Active). Try rephrasing your question "
"or disabling Strict Mode.</i>"
)

SessionManager.add_message("assistant", response_html)
st.rerun()


# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================


def main() -> None:
SessionManager.initialize()

chunk_size, overlap, top_k, strict_mode = render_sidebar()

    st.title("🧠 DocuChat Enterprise Demo")
    st.title("📝 DocuChat Demo")
st.caption(
"Local Retrieval-Augmented Generation system for PDF document analysis."
)

tab1, tab2, tab3 = st.tabs(
["📤 Upload & Process", "🔎 Research Interface", "📈 System Analytics"]
)

with tab1:
render_file_upload_area(chunk_size, overlap)

with tab2:
render_chat_interface(top_k, strict_mode)

with tab3:
render_analytics_tab(st.session_state.db)


main()
~

