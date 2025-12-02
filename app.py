"""
DocuChat Enterprise Edition - Ultimate
--------------------------------------
A production-grade RAG (Retrieval-Augmented Generation) system.
Includes Unit Testing, System Monitoring, and Advanced Analytics.

Author: DocuChat Engineering Team
Version: 4.0.0-stable
License: Proprietary / MIT
"""

import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import re
import logging
import random
import psutil  # Sistem kaynaklarını izlemek için (Opsiyonel, hata verirse try-except var)
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

# ==============================================================================
# 1. SYSTEM CONFIGURATION & GLOBAL STYLES
# ==============================================================================

st.set_page_config(
    page_title="DocuChat Enterprise Architect",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit',
        'About': "DocuChat Enterprise v4.0\nPowered by FAISS & Transformer Models"
    }
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Global Font & Background */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }

    /* Chat Bubbles */
    .user-msg {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        padding: 15px;
        border-radius: 12px 12px 0 12px;
        margin: 10px 0 10px 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .bot-msg {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        color: #374151;
        padding: 20px;
        border-radius: 12px 12px 12px 0;
        margin: 10px 20% 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Source Cards */
    .source-card {
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
        padding: 10px;
        margin-top: 8px;
        font-size: 0.9em;
    }
    .badge {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: bold;
        color: white;
    }
    .badge-high { background-color: #16a34a; }
    .badge-med { background-color: #ca8a04; }
    .badge-low { background-color: #dc2626; }

    /* Status Indicators */
    .status-ok { color: #16a34a; font-weight: bold; }
    .status-warn { color: #ca8a04; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. LOGGING & MONITORING SUBSYSTEM
# ==============================================================================

class SystemLogger:
    """
    Centralized logging system with UI display capabilities.
    """
    def _init_(self):
        self.logs = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DocuChat")

    def log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.logs.append(entry)
        
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "error":
            self.logger.error(message)
        elif level.lower() == "warning":
            self.logger.warning(message)

    def get_logs(self) -> str:
        return "\n".join(self.logs[-50:]) # Return last 50 logs


class SystemMonitor:
    """
    Simulates system resource monitoring (CPU/RAM).
    """
    @staticmethod
    def get_stats():
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            return cpu, ram
        except:
            # Fallback if psutil is not installed
            return 0.0, 0.0

# Initialize in Session
if 'logger' not in st.session_state:
    st.session_state.logger = SystemLogger()

logger = st.session_state.logger


# ==============================================================================
# 3. DATA MODELS
# ==============================================================================

@dataclass
class DocumentChunk:
    """Schema for a single unit of text."""
    id: str
    text: str
    source_file: str
    page_number: int
    word_count: int
    char_count: int

@dataclass
class SearchResult:
    """Schema for retrieval results."""
    chunk: DocumentChunk
    score: float
    confidence_level: str


# ==============================================================================
# 4. ETL PIPELINE (Extract, Transform, Load)
# ==============================================================================

class TextCleaner:
    """Advanced Regex-based text sanitization."""
    
    @staticmethod
    def clean(text: str) -> str:
        if not text: return ""
        
        # Decode/Encode to fix weird unicode
        text = text.encode("utf-8", "ignore").decode("utf-8")
        
        # Remove PDF artifacts
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix hyphens at line breaks (e.g. "com- puter")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalize whitespace
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

class DocumentProcessor:
    """Handles PDF ingestion and chunking strategies."""
    
    def _init_(self):
        self.cleaner = TextCleaner()
    
    def process(self, file_obj, chunk_size=500, overlap=50) -> List[DocumentChunk]:
        chunks = []
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            file_name = file_obj.name
            
            logger.log("info", f"Processing start: {file_name}")

            for page_idx, page in enumerate(pdf_reader.pages):
                raw_text = page.extract_text()
                clean_text = self.cleaner.clean(raw_text)
                
                if not clean_text: continue

                words = clean_text.split()
                
                # Sliding Window Algorithm
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i : i + chunk_size]
                    
                    if len(chunk_words) < 15: continue # Skip noise
                    
                    chunk_text = " ".join(chunk_words)
                    
                    chunks.append(DocumentChunk(
                        id=f"{file_name}p{page_idx+1}{i}",
                        text=chunk_text,
                        source_file=file_name,
                        page_number=page_idx + 1,
                        word_count=len(chunk_words),
                        char_count=len(chunk_text)
                    ))
            
            logger.log("info", f"Processing complete: {len(chunks)} chunks created.")
            return chunks
            
        except Exception as e:
            logger.log("error", f"Failed to process {file_obj.name}: {e}")
            return []


# ==============================================================================
# 5. CORE AI ENGINE (Vector DB)
# ==============================================================================

class VectorEngine:
    """
    Manages Embeddings and FAISS Index operations.
    """
    def _init_(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks_map = {}
        self.is_ready = False

    def initialize_model(self):
        if self.model is None:
            logger.log("info", f"Loading AI Model: {self.model_name}")
            with st.spinner("Booting Neural Engine..."):
                self.model = SentenceTransformer(self.model_name)

    def ingest(self, chunks: List[DocumentChunk]):
        self.initialize_model()
        
        if not chunks:
            logger.log("warning", "No chunks to ingest.")
            return False

        texts = [c.text for c in chunks]
        
        logger.log("info", f"Vectorizing {len(chunks)} text segments...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Map ID to Object
        self.chunks_map = {i: c for i, c in enumerate(chunks)}
        self.is_ready = True
        return True

    def search(self, query: str, top_k=3) -> List[SearchResult]:
        if not self.is_ready: return []
        
        self.initialize_model()
        query_vec = self.model.encode([query]).astype('float32')
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            chunk = self.chunks_map.get(idx)
            score = distances[0][i]
            
            # Confidence Logic
            if score < 0.9: conf = "HIGH"
            elif score < 1.3: conf = "MEDIUM"
            else: conf = "LOW"
            
            results.append(SearchResult(chunk, score, conf))
            
        return results

    def get_visualization_data(self):
        """Prepares data for 3D visualization"""
        if not self.is_ready: return None
        
        # Re-encoding for visualization (simplified for memory safety)
        # In prod, cache these.
        sample_chunks = list(self.chunks_map.values())[:500] # Limit to 500 for UI speed
        texts = [c.text for c in sample_chunks]
        meta = [f"Page {c.page_number}" for c in sample_chunks]
        
        embeddings = self.model.encode(texts)
        return embeddings, texts, meta


# ==============================================================================
# 6. UNIT TESTING SUITE (Professional Feature)
# ==============================================================================

class UnitTestEngine:
    """
    Self-diagnostic tools to ensure system integrity.
    """
    def run_diagnostics(self):
        results = []
        
        # Test 1: Library Check
        try:
            import torch
            results.append(("PyTorch Check", "PASS", "Library found"))
        except:
            results.append(("PyTorch Check", "WARN", "Using CPU only"))

        # Test 2: Model Loading
        try:
            test_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            vec = test_model.encode(["test"])
            if vec.shape[1] == 384:
                results.append(("Model Integrity", "PASS", "Dimensions correct (384)"))
            else:
                results.append(("Model Integrity", "FAIL", "Dimensions mismatch"))
        except Exception as e:
            results.append(("Model Loading", "FAIL", str(e)))

        # Test 3: Session State
        if 'db' in st.session_state:
            results.append(("Session Persistence", "PASS", "DB initialized"))
        else:
            results.append(("Session Persistence", "WARN", "Not initialized"))

        return results


# ==============================================================================
# 7. UI COMPONENTS & MANAGERS
# ==============================================================================

def render_sidebar():
    """Renders the settings sidebar."""
    with st.sidebar:
        st.title("🎛 Control Panel")
        st.markdown("---")
        
        st.subheader("1. System Status")
        cpu, ram = SystemMonitor.get_stats()
        st.progress(cpu / 100, text=f"CPU Usage: {cpu}%")
        st.progress(ram / 100, text=f"RAM Usage: {ram}%")
        
        st.markdown("---")
        
        st.subheader("2. Ingestion Config")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500)
        overlap = st.slider("Overlap", 0, 200, 50)
        
        st.subheader("3. Search Config")
        top_k = st.slider("Top K Results", 1, 10, 3)
        strict_mode = st.toggle("Strict Mode (Filter Low Confidence)", value=True)
        
        st.markdown("---")
        
        if st.button("Run Unit Tests 🛠"):
            st.session_state.show_tests = True
            
        with st.expander("Live Logs"):
            st.code(st.session_state.logger.get_logs(), language="text")
            
    return chunk_size, overlap, top_k, strict_mode


def render_analytics(db: VectorEngine):
    """Renders the Analytics Dashboard."""
    st.header("📈 Enterprise Analytics")
    
    if not db.is_ready:
        st.warning("No data loaded.")
        return

    # 1. Word Frequency Analysis
    all_text = " ".join([c.text for c in db.chunks_map.values()])
    words = [w.lower() for w in all_text.split() if len(w) > 4] # Filter short words
    counts = Counter(words).most_common(15)
    
    df_words = pd.DataFrame(counts, columns=['Word', 'Count'])
    fig_words = px.bar(df_words, x='Word', y='Count', title="Top Keywords in Document", color='Count')
    
    # 2. 3D Vector Space
    embeddings, texts, meta = db.get_visualization_data()
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    vecs_3d = pca.fit_transform(embeddings)
    df_3d = pd.DataFrame(vecs_3d, columns=['x', 'y', 'z'])
    df_3d['meta'] = meta
    df_3d['text'] = [t[:50]+"..." for t in texts]
    
    fig_3d = px.scatter_3d(df_3d, x='x', y='y', z='z', color='meta', hover_name='text', title="Semantic Cluster Map")

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig_words, use_container_width=True)
    with col2: st.plotly_chart(fig_3d, use_container_width=True)


def render_chat(top_k, strict_mode):
    """Renders the Chat Interface."""
    st.subheader("💬 Knowledge Assistant")
    
    if not st.session_state.db.is_ready:
        st.info("👈 Please upload and process a document in the 'Upload' tab.")
        return

    # History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.markdown(f"<div class='user-msg'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # Input
    prompt = st.chat_input("Ask about your documents...")
    
    if prompt:
        # User Logic
        st.session_state.history.append({"role": "user", "content": prompt})
        st.rerun()

    # Bot Logic
    if st.session_state.history and st.session_state.history[-1]['role'] == 'user':
        last_prompt = st.session_state.history[-1]['content']
        
        with st.spinner("Analyzing..."):
            results = st.session_state.db.search(last_prompt, top_k)
            
            response_html = ""
            valid_results = 0
            
            if not results:
                response_html = "No information found."
            else:
                response_html += f"<b>Found {len(results)} references:</b><br>"
                
                for res in results:
                    if strict_mode and res.confidence_level == "LOW":
                        continue
                        
                    valid_results += 1
                    badge_cls = f"badge-{res.confidence_level.lower()}" if res.confidence_level != "MEDIUM" else "badge-med"
                    
                    response_html += f"""
                    <div class="source-card">
                        <div style="margin-bottom:5px;">
                            📄 {res.chunk.source_file} (Page {res.chunk.page_number})
                            <span class="badge {badge_cls}">{res.confidence_level}</span>
                        </div>
                        <i>"{res.chunk.text}"</i>
                    </div>
                    """
                
                if strict_mode and valid_results == 0:
                    response_html = "<i>Matches found, but they were filtered out due to low confidence (Strict Mode).</i>"

            st.session_state.history.append({"role": "bot", "content": response_html})
            st.rerun()


# ==============================================================================
# 8. MAIN EXECUTION FLOW
# ==============================================================================

def main():
    # Initialize Session
    if 'db' not in st.session_state:
        st.session_state.db = VectorEngine()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'show_tests' not in st.session_state:
        st.session_state.show_tests = False

    # Sidebar
    chunk_size, overlap, top_k, strict_mode = render_sidebar()

    # Unit Test Modal
    if st.session_state.show_tests:
        st.info("Running System Diagnostics...")
        tester = UnitTestEngine()
        results = tester.run_diagnostics()
        for name, status, msg in results:
            icon = "✅" if status == "PASS" else "⚠" if status == "WARN" else "❌"
            st.write(f"{icon} *{name}*: {msg}")
        if st.button("Close Tests"):
            st.session_state.show_tests = False
            st.rerun()
        st.divider()

    # Main Tabs
    st.title("🧠 DocuChat Enterprise")
    tab1, tab2, tab3 = st.tabs(["📤 Ingestion", "🔎 Chat", "📊 Analytics"])

    with tab1:
        st.header("Document Ingestion")
        files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
        
        if files:
            if st.button("Process Files 🚀"):
                processor = DocumentProcessor()
                all_chunks = []
                
                bar = st.progress(0, text="Processing...")
                for i, f in enumerate(files):
                    chunks = processor.process(f, chunk_size, overlap)
                    all_chunks.extend(chunks)
                    bar.progress((i + 1) / len(files))
                
                if all_chunks:
                    st.session_state.db.ingest(all_chunks)
                    st.success(f"Ingested {len(all_chunks)} chunks successfully!")
                    time.sleep(1)
                    st.rerun()

    with tab2:
        render_chat(top_k, strict_mode)
        
    with tab3:
        render_analytics(st.session_state.db)

if _name_ == "_main_":
    main()
