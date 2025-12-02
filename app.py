import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import io
import os
import pickle
from typing import List, Tuple
import plotly.express as px
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CORE CLASSES (OOP Architecture)
# -----------------------------------------------------------------------------

class DocumentProcessor:
    """Handles PDF reading, text cleaning, and chunking."""
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
        """Extracts raw text and page count from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            num_pages = len(pdf_reader.pages)
            
            # Using a list and joining at the end is faster for large texts (9000+ words)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text_parts.append(content)
            
            full_text = "\n".join(text_parts)
            return full_text, num_pages
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return "", 0

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans whitespace and standardizes text."""
        # Replace newlines with spaces to maintain flow in chunks
        text = text.replace("\n", " ")
        # Remove multiple spaces
        text = " ".join(text.split())
        return text

    @staticmethod
    def create_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits text into overlapping chunks.
        - chunk_size: Number of words per chunk.
        - overlap: Number of words to overlap between chunks (preserves context).
        """
        words = text.split()
        chunks = []
        if not words:
            return chunks
            
        # Sliding window approach
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks


class VectorStoreManager:
    """Manages the AI model, embeddings, and FAISS index."""
    
    def _init_(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None
        self.index = None
        self.chunks = []
        self.embeddings = None

    @property
    def model(self):
        """Lazy loads the model only when needed."""
        if self._model is None:
            with st.spinner(f"Loading AI Model ({self.model_name})..."):
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def process_and_index(self, chunks: List[str]):
        """Generates embeddings and builds the FAISS index."""
        self.chunks = chunks
        if not chunks:
            return
        
        # Encode all chunks (this might take time for 9000+ words, hence the progress bar)
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS Index (L2 Distance / Euclidean)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        return self.index

    def search(self, query: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """Searches for the top-k most relevant chunks."""
        if not self.index:
            return [], []
            
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        scores = []
        
        # indices[0] contains the IDs of the matching chunks
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                scores.append(distances[0][i])
                
        return results, scores

    def save_index(self, file_path: str = "vector_store.pkl"):
        """Saves the current index and chunks to disk."""
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, file_path: str = "vector_store.pkl"):
        """Loads index and chunks from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError("No saved index found.")
            
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        
        # Rebuild FAISS index from loaded embeddings
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)


class VisualizationEngine:
    """Handles 2D plotting of high-dimensional vectors."""
    
    @staticmethod
    def plot_embeddings_2d(embeddings, chunks, query_embedding=None):
        """Reduces vectors to 2D using PCA and plots them with Plotly."""
        if embeddings is None or len(embeddings) < 3:
            st.warning("Not enough data to visualize (needs at least 3 chunks).")
            return None

        # Dimensionality Reduction (384 dim -> 2 dim)
        pca = PCA(n_components=2)
        reduced_vecs = pca.fit_transform(embeddings)
        
        df = pd.DataFrame(reduced_vecs, columns=['x', 'y'])
        # Preview first 100 chars for tooltip
        df['text_preview'] = [c[:120] + "..." for c in chunks]
        df['type'] = 'Document Segment'
        
        # Add query point if it exists
        if query_embedding is not None:
            query_reduced = pca.transform(query_embedding)
            df_query = pd.DataFrame(query_reduced, columns=['x', 'y'])
            df_query['text_preview'] = '🔴 YOUR QUESTION'
            df_query['type'] = 'Current Question'
            df = pd.concat([df, df_query], ignore_index=True)

        fig = px.scatter(
            df, x='x', y='y', 
            color='type', 
            hover_data=['text_preview'],
            title="Semantic Data Map (PCA Projection)",
            color_discrete_map={'Document Segment': '#0084ff', 'Current Question': '#ff4b4b'},
            size_max=12
        )
        fig.update_layout(
            plot_bgcolor='rgba(240,242,246,0.5)',
            xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
            yaxis=dict(showgrid=True, gridcolor='#e0e0e0')
        )
        return fig


# -----------------------------------------------------------------------------
# 3. MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("🎛 Control Panel")
        
        st.subheader("1. File Management")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", help="Supports large text-based PDFs.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Index 💾"):
                if st.session_state.get('vector_manager') and st.session_state.get('processed'):
                    try:
                        st.session_state['vector_manager'].save_index()
                        st.success("Saved!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Process a file first.")
        with col2:
            if st.button("Load Index 📂"):
                try:
                    if 'vector_manager' not in st.session_state:
                        st.session_state['vector_manager'] = VectorStoreManager()
                    st.session_state['vector_manager'].load_index()
                    st.session_state['processed'] = True
                    st.session_state['file_name'] = "Loaded from Disk"
                    st.success("Loaded!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")

        st.divider()
        st.subheader("2. Analysis Parameters")
        chunk_size = st.slider("Chunk Size (Words)", 100, 2000, 500, help="Larger chunks = more context, but less specific.")
        overlap = st.slider("Overlap (Words)", 0, 500, 50, help="Prevents cutting sentences in half.")
        top_k = st.slider("Retrieval Count (Top-K)", 1, 20, 4, help="How many sections to retrieve.")
        
        st.divider()
        if st.button("Clear History 🗑"):
            st.session_state.messages = []
            st.rerun()

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Enterprise Document Assistant. Upload a large PDF to begin."}]
    
    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager()
        
    if "processed" not in st.session_state:
        st.session_state.processed = False

    # --- File Processing Logic ---
    if uploaded_file and not st.session_state.processed:
        with st.status("🚀 Processing large document...", expanded=True) as status:
            st.write("📖 Reading PDF content...")
            file_bytes = uploaded_file.read()
            text_content, num_pages = DocumentProcessor.extract_text_from_pdf(file_bytes)
            
            # Basic validation for content
            if len(text_content) < 50:
                status.update(label="Error: PDF is empty or unreadable.", state="error")
                st.error("Could not extract text. Ensure the PDF is not a scanned image.")
                return

            st.write(f"🧹 Cleaning and analyzing {len(text_content.split())} words...")
            clean_content = DocumentProcessor.clean_text(text_content)
            
            st.write("✂ Splitting into semantic chunks...")
            chunks = DocumentProcessor.create_chunks(clean_content, chunk_size, overlap)
            
            st.write(f"🧠 Generating vector embeddings for {len(chunks)} chunks...")
            st.session_state.vector_manager.process_and_index(chunks)
            
            # Save metadata to session
            st.session_state.processed = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.num_pages = num_pages
            st.session_state.total_words = len(clean_content.split())
            st.session_state.total_chunks = len(chunks)
            
            status.update(label="✅ System Ready!", state="complete")

    # --- Main Interface ---
    st.title("🧠 DocuChat Enterprise")
    
    # Show document stats if processed
    if st.session_state.processed:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("File Name", st.session_state.get('file_name', 'N/A'))
        c2.metric("Total Pages", st.session_state.get('num_pages', 0))
        c3.metric("Total Words", f"{st.session_state.get('total_words', 0):,}")
        c4.metric("Knowledge Chunks", st.session_state.get('total_chunks', 0))

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Visualization", "📄 Data Explorer"])

    # ---------------- TAB 1: CHAT ----------------
    with tab1:
        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        # Chat Input
        if prompt := st.chat_input("Ask a question about the document..."):
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Assistant Logic
            with st.chat_message("assistant"):
                vm = st.session_state.vector_manager
                
                if not vm.index:
                    st.warning("Please upload a document first.")
                else:
                    # 1. Search
                    results, scores = vm.search(prompt, k=top_k)
                    
                    if not results:
                        response_text = "I couldn't find any relevant information in the document."
                    else:
                        # 2. Format Response
                        response_text = f"*I found {len(results)} relevant sections in '{st.session_state.get('file_name', 'the document')}':*\n\n"
                        
                        for i, (res, score) in enumerate(zip(results, scores), 1):
                            # Interpret score (Lower L2 distance = better match)
                            match_quality = "Excellent" if score < 0.8 else "Good" if score < 1.2 else "Weak"
                            color = "#28a745" if match_quality == "Excellent" else "#ffc107" if match_quality == "Good" else "#dc3545"
                            
                            response_text += f"""
<div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {color};">
    <div style="font-weight: bold; color: #333; margin-bottom: 4px;">
        Result #{i} <span style="font-size: 0.8em; color: #666; font-weight: normal;">(Match: {match_quality}, Score: {score:.2f})</span>
    </div>
    <div style="color: #444; line-height: 1.5;">{res}</div>
</div>
"""
                        response_text += "\n<small style='color:grey'>Note: These are exact extracts from the document. No external AI generation was used.</small>"

                # Display and Save
                st.markdown(response_text, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    # ---------------- TAB 2: VISUALIZATION ----------------
    with tab2:
        st.header("Semantic Data Map")
        st.info("This interactive map shows how your document is distributed in the 'meaning space'. Dots close together have similar meanings.")
        
        if st.button("Generate/Refresh Map"):
            with st.spinner("Calculating PCA projection..."):
                vm = st.session_state.vector_manager
                if vm.embeddings is not None:
                    # Get user's last question vector if available
                    last_query_vec = None
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                         last_query_text = st.session_state.messages[-1]["content"]
                         last_query_vec = vm.model.encode([last_query_text])
                    
                    fig = VisualizationEngine.plot_embeddings_2d(vm.embeddings, vm.chunks, last_query_vec)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available. Upload a file first.")

    # ---------------- TAB 3: DATA EXPLORER ----------------
    with tab3:
        st.header("Raw Data Inspector")
        st.write("Review how the system split your document.")
        
        vm = st.session_state.vector_manager
        if vm.chunks:
            df_chunks = pd.DataFrame(vm.chunks, columns=["Chunk Content"])
            df_chunks.index.name = "ID"
            st.dataframe(df_chunks, use_container_width=True)
            
            st.markdown("### Detailed View")
            selected_id = st.number_input("Enter Chunk ID to inspect:", min_value=0, max_value=len(vm.chunks)-1, value=0)
            st.text_area("Full Content:", value=vm.chunks[selected_id], height=200)
        else:
            st.info("No data processed yet.")

if _name_ == "_main_":
    main()
