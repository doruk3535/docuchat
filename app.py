"""
DocuChat Enterprise Edition
---------------------------
A local, secure, and robust RAG (Retrieval-Augmented Generation) system 
designed for high-volume document analysis without external API dependencies.

Author: DocuChat Team
Version: 3.5.0 (Release Candidate)
License: MIT
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
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# ==============================================================================
# 1. SYSTEM CONFIGURATION & LOGGING SETUP
# ==============================================================================

# Configure Streamlit Page
st.set_page_config(
    page_title="DocuChat Enterprise System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit',
        'Report a bug': "https://github.com/streamlit",
        'About': "# DocuChat Enterprise\nLocal RAG System v3.5"
    }
)

# Setup Custom Logging
class SystemLogger:
    """
    Handles application-wide logging to display in the UI and console.
    """
    def _init_(self):
        self.logs = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DocuChat")

    def info(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [INFO] {message}"
        self.logs.append(entry)
        self.logger.info(message)

    def error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [ERROR] {message}"
        self.logs.append(entry)
        self.logger.error(message)

    def warning(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [WARN] {message}"
        self.logs.append(entry)
        self.logger.warning(message)

    def get_logs(self) -> str:
        return "\n".join(self.logs)

# Initialize Logger in Session State
if 'sys_logger' not in st.session_state:
    st.session_state.sys_logger = SystemLogger()

logger = st.session_state.sys_logger

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Main Container */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Headers */
    h1 { color: #1e3a8a; font-weight: 700; }
    h2 { color: #1e40af; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
    h3 { color: #3b82f6; }

    /* Chat Messages */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .user-bubble {
        align_self: flex-end;
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        padding: 15px;
        border-radius: 15px 15px 0 15px;
        margin-left: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .bot-bubble {
        align_self: flex-start;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        color: #374151;
        padding: 20px;
        border-radius: 15px 15px 15px 0;
        margin-right: 10%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Source Box */
    .source-metadata {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .source-content {
        font-family: 'Georgia', serif;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin-top: 5px;
        background-color: #f9fafb;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #2563eb;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA STRUCTURES (Dataclasses)
# ==============================================================================

@dataclass
class DocumentChunk:
    """
    Represents a single piece of processed text with its metadata.
    """
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
    """
    Represents a retrieved result from the vector store.
    """
    chunk: DocumentChunk
    score: float
    relevance_label: str


# ==============================================================================
# 3. TEXT PROCESSING ENGINE (ETL Layer)
# ==============================================================================

class TextProcessor:
    """
    Advanced text cleaning and normalization utilities.
    Includes Regex patterns to fix common PDF parsing errors.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Applies a pipeline of cleaning operations to raw text.
        """
        if not text:
            return ""

        # 1. Normalize unicode characters
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # 2. Remove page headers/footers (e.g., "Page 1 of 20", "Confidential")
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE) # Standalone page numbers

        # 3. Fix broken hyphenation (e.g., "environ- ment" -> "environment")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 4. Remove excessive newlines that break sentences
        text = text.replace('\n', ' ')

        # 5. Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)

        # 6. Remove very short nonsense segments (optional)
        if len(text) < 5:
            return ""

        return text.strip()

    @staticmethod
    def calculate_stats(text: str) -> Dict[str, Any]:
        """Calculates linguistic statistics for the text."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_len": sum(len(w) for w in words) / len(words) if words else 0
        }


class DocumentIngestor:
    """
    Handles file upload, reading, and chunking logic.
    """
    def _init_(self):
        self.processor = TextProcessor()

    def process_pdf(self, file_obj, chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """
        Reads a PDF file and splits it into semantic chunks.
        """
        chunks = []
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            file_name = file_obj.name
            
            logger.info(f"Started processing file: {file_name} ({len(pdf_reader.pages)} pages)")

            for page_idx, page in enumerate(pdf_reader.pages):
                raw_text = page.extract_text()
                clean_text = self.processor.clean_text(raw_text)
                
                if not clean_text:
                    continue

                # Sliding Window Chunking
                words = clean_text.split()
                if not words:
                    continue
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i : i + chunk_size]
                    chunk_str = " ".join(chunk_words)
                    
                    # Minimum viability check
                    if len(chunk_words) > 10:
                        chunk_obj = DocumentChunk(
                            id=f"{file_name}p{page_idx+1}{i}",
                            text=chunk_str,
                            source_file=file_name,
                            page_number=page_idx + 1,
                            char_count=len(chunk_str),
                            word_count=len(chunk_words)
                        )
                        chunks.append(chunk_obj)
            
            logger.info(f"Finished processing {file_name}. Generated {len(chunks)} chunks.")
            return chunks

        except Exception as e:
            logger.error(f"Error reading PDF {file_obj.name}: {str(e)}")
            st.error(f"Failed to process {file_obj.name}. See logs for details.")
            return []


# ==============================================================================
# 4. VECTOR DATABASE & SEARCH ENGINE (Backend)
# ==============================================================================

class VectorDatabase:
    """
    Manages FAISS index and Embedding Model.
    """
    def _init_(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks_map: Dict[int, DocumentChunk] = {} # Maps FAISS ID to Chunk Object
        self.is_initialized = False

    def load_model(self):
        """Lazy loader for the heavy transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            with st.spinner(f"Initializing AI Core ({self.model_name})..."):
                self.model = SentenceTransformer(self.model_name)
    
    def build_index(self, chunks: List[DocumentChunk]):
        """
        Converts text chunks into vectors and builds the FAISS index.
        """
        self.load_model()
        
        if not chunks:
            logger.warning("No chunks provided to build index.")
            return False

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Prepare text list
        texts = [c.text for c in chunks]
        
        # Generate Embeddings (Batch processing happens internally)
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Initialize FAISS (L2 Distance / Euclidean)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Map indices to objects
        self.chunks_map = {i: chunk for i, chunk in enumerate(chunks)}
        self.is_initialized = True
        
        logger.info("Vector index built successfully.")
        return True

    def query(self, query_text: str, top_k: int = 3) -> List[SearchResult]:
        """
        Searches the database for relevant chunks.
        Includes Logic to determine relevance labels.
        """
        if not self.is_initialized:
            logger.error("Attempted to query uninitialized database.")
            return []

        self.load_model()
        
        # Vectorize Query
        query_vec = self.model.encode([query_text]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # No match found
            
            score = distances[0][i]
            chunk = self.chunks_map.get(idx)
            
            if chunk:
                # RELEVANCE LOGIC (L2 Distance)
                # Lower is better. 
                # < 0.8: Excellent
                # 0.8 - 1.2: Good
                # > 1.2: Weak / Noise
                
                label = "LOW"
                if score < 0.90:
                    label = "HIGH"
                elif score < 1.30:
                    label = "MEDIUM"
                
                results.append(SearchResult(chunk=chunk, score=score, relevance_label=label))

        return results
    
    def get_embeddings_for_viz(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Helper to export data for 3D visualization."""
        if not self.is_initialized:
            return None, [], []
        
        # We need to regenerate or store embeddings. 
        # For memory efficiency, we usually don't store raw embeddings in RAM for huge datasets,
        # but for visualization we will re-encode a subset or assume we kept them if we modified the class.
        # Here we will re-encode texts for the plot to be safe (or store them in build_index if memory allows).
        # To keep it fast, let's assume we re-encode just for the plot.
        
        texts = [c.text[:50] + "..." for c in self.chunks_map.values()]
        sources = [c.source_file for c in self.chunks_map.values()]
        
        # Note: In a real prod app, you would cache 'embeddings' in self.embeddings
        full_texts = [c.text for c in self.chunks_map.values()]
        embeddings = self.model.encode(full_texts, show_progress_bar=False)
        
        return embeddings, texts, sources


# ==============================================================================
# 5. SESSION STATE MANAGEMENT
# ==============================================================================

class SessionManager:
    """
    Centralized management of Streamlit Session State.
    """
    @staticmethod
    def initialize():
        if 'db' not in st.session_state:
            st.session_state.db = VectorDatabase(model_name='sentence-transformers/all-MiniLM-L6-v2')
        if 'ingestor' not in st.session_state:
            st.session_state.ingestor = DocumentIngestor()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        if 'stats' not in st.session_state:
            st.session_state.stats = {"total_words": 0, "total_chunks": 0}

    @staticmethod
    def add_message(role: str, content: str):
        st.session_state.chat_history.append({
            "role": role, 
            "content": content, 
            "timestamp": datetime.now().isoformat()
        })

    @staticmethod
    def clear_history():
        st.session_state.chat_history = []
        logger.info("Chat history cleared by user.")


# ==============================================================================
# 6. USER INTERFACE COMPONENTS
# ==============================================================================

def render_sidebar():
    """Renders the configuration sidebar."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9626/9626620.png", width=80)
        st.title("System Control")
        st.caption(f"v3.5.0 | {datetime.now().strftime('%Y-%m-%d')}")
        
        st.divider()
        
        st.subheader("1. Ingestion Settings")
        chunk_size = st.slider(
            "Chunk Size (Tokens)", 
            min_value=100, 
            max_value=1000, 
            value=500,
            help="Number of words per semantic unit. Smaller = More specific."
        )
        overlap = st.slider(
            "Overlap Window", 
            min_value=0, 
            max_value=200, 
            value=50,
            help="Words shared between chunks to preserve context."
        )
        
        st.divider()
        
        st.subheader("2. Search Parameters")
        top_k = st.slider("Retrieval Depth (K)", 1, 10, 3)
        strict_mode = st.checkbox("Strict Relevance Filter", value=True, help="Hides low confidence results.")
        
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


def render_analytics_tab(db: VectorDatabase):
    """Renders the Analytics Dashboard using Plotly."""
    st.header("📊 Knowledge Base Analytics")
    
    if not db.is_initialized:
        st.info("No data available. Please upload and process documents first.")
        return

    # 1. High-Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", st.session_state.stats['total_chunks'])
    col2.metric("Total Words", f"{st.session_state.stats['total_words']:,}")
    col3.metric("Processed Files", len(st.session_state.processed_files))
    col4.metric("Embedding Dim", "384")

    st.markdown("---")

    # 2. 3D Vector Space Visualization
    st.subheader("Semantic Vector Space (PCA Reduced)")
    with st.spinner("Calculating 3D Projection..."):
        embeddings, labels, sources = db.get_embeddings_for_viz()
        
        if embeddings is not None and len(embeddings) > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            projections = pca.fit_transform(embeddings)
            
            df_viz = pd.DataFrame(projections, columns=['x', 'y', 'z'])
            df_viz['text'] = labels
            df_viz['source'] = sources
            
            fig = px.scatter_3d(
                df_viz, x='x', y='y', z='z',
                color='source',
                hover_data=['text'],
                title="3D Document Cluster Map",
                template="plotly_white",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data points for 3D visualization.")

    # 3. Chunk Length Distribution
    st.subheader("Chunk Length Distribution")
    chunk_lengths = [c.word_count for c in db.chunks_map.values()]
    fig_hist = px.histogram(x=chunk_lengths, nbins=20, labels={'x': 'Word Count'}, title="Distribution of Information Density")
    st.plotly_chart(fig_hist, use_container_width=True)


def render_file_upload_area(chunk_size, overlap):
    """Renders the file upload section."""
    st.subheader("📂 Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Files (Multi-file support)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Check if new files need processing
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            if st.button(f"Process {len(new_files)} New Files", type="primary"):
                all_chunks = []
                
                # Progress Bar
                progress_bar = st.progress(0, text="Starting ingestion...")
                
                for i, file in enumerate(new_files):
                    chunks = st.session_state.ingestor.process_pdf(file, chunk_size, overlap)
                    all_chunks.extend(chunks)
                    st.session_state.processed_files.add(file.name)
                    progress_bar.progress((i + 1) / len(new_files), text=f"Processed {file.name}")

                if all_chunks:
                    # If DB exists, we might want to extend it, but for simplicity we rebuild
                    # (FAISS allows adding, but we rebuild here for cleaner state in this demo)
                    
                    # Merge with existing chunks if any (Advanced feature, simplified here)
                    # For this demo, we rebuild the index with ALL currently uploaded files
                    # To do that properly, we should re-read all uploaded_files. 
                    # But Streamlit retains file objects.
                    
                    # Let's collect ALL chunks from ALL uploaded files to be safe
                    total_chunks_collection = []
                    for f in uploaded_files:
                        chunks = st.session_state.ingestor.process_pdf(f, chunk_size, overlap)
                        total_chunks_collection.extend(chunks)

                    success = st.session_state.db.build_index(total_chunks_collection)
                    
                    if success:
                        st.session_state.stats['total_chunks'] = len(total_chunks_collection)
                        st.session_state.stats['total_words'] = sum(c.word_count for c in total_chunks_collection)
                        st.success(f"Successfully indexed {len(total_chunks_collection)} chunks!")
                        time.sleep(1)
                        st.rerun()
                
                progress_bar.empty()
        else:
            st.success("All uploaded files are processed and ready.")


def render_chat_interface(top_k, strict_mode):
    """Renders the main chat UI."""
    st.subheader("💬 AI Research Assistant")
    
    # Check readiness
    if not st.session_state.db.is_initialized:
        st.warning("⚠ System Offline. Please process documents in the 'Upload' tab.")
        return

    # Display History
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"<div class='user-bubble'>👤 <b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>🤖 <b>System:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat Input
    prompt = st.chat_input("Ask a specific question about your documents...")
    
    if prompt:
        # User Message
        SessionManager.add_message("user", prompt)
        st.rerun() # Immediate visual update

    # Handle Assistant Response (if last message was user)
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
        last_query = st.session_state.chat_history[-1]['content']
        
        with st.chat_message("assistant"): # Use st spinner placeholder
            with st.spinner("Analyzing semantic vectors..."):
                time.sleep(0.3) # UI feel
                results = st.session_state.db.query(last_query, top_k=top_k)
                
                # HTML Builder for response
                response_html = ""
                
                found_valid_result = False
                
                if not results:
                    response_html = "<i>No relevant information found in the provided documents.</i>"
                else:
                    response_html += f"<div style='margin-bottom:10px;'><b>Found {len(results)} references:</b></div>"
                    
                    for res in results:
                        # STRICT MODE FILTER
                        if strict_mode and res.relevance_label == "LOW":
                            continue
                        
                        found_valid_result = True
                        
                        # Style based on score
                        color = "#16a34a" if res.relevance_label == "HIGH" else "#ca8a04" if res.relevance_label == "MEDIUM" else "#dc2626"
                        
                        response_html += f"""
                        <div style="border-left: 5px solid {color}; padding-left: 15px; margin-bottom: 20px; background: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
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
                         response_html = "<i>Matching results were found but filtered out due to low relevance (Strict Mode Active). Try rephrasing your question or disabling Strict Mode.</i>"

                # Save and Display
                SessionManager.add_message("assistant", response_html)
                st.rerun()


# ==============================================================================
# 7. MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    # 1. Initialize State
    SessionManager.initialize()
    
    # 2. Render Sidebar & Get Settings
    chunk_size, overlap, top_k, strict_mode = render_sidebar()
    
    # 3. Main Title
    st.title("🧠 DocuChat Enterprise")
    st.caption("Advanced Retrieval-Augmented Generation (RAG) System for Document Analysis")
    
    # 4. Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔎 Research Interface", "📈 System Analytics"])
    
    with tab1:
        render_file_upload_area(chunk_size, overlap)
    
    with tab2:
        render_chat_interface(top_k, strict_mode)
        
    with tab3:
        render_analytics_tab(st.session_state.db)

if _name_ == "_main_":
    main()
