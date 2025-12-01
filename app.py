import io
from typing import List, Tuple

import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------
# ÜCRETSİZ DOCUCHAT
# - OpenAI / API KEY YOK
# - Embedding: all-MiniLM-L6-v2 (sentence-transformers)
# - Retrieval: FAISS (cosine similarity)
# - Answer: En alakalı cümlelerden extractive özet
# -----------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embedder():
    """SentenceTransformer modelini bir kez yükleyip cache'ler."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# -------------------------
# DOCUMENT PROCESSING LAYER
# -------------------------
def parse_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = " ".join(lines)
    cleaned = " ".join(cleaned.split())
    return cleaned


def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


# -------------------------
# VECTOR SEARCH LAYER
# (Embedding + FAISS Index)
# -------------------------
def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    sentence-transformers ile embedding üretir ve
    cosine similarity için vektörleri normalize eder.
    """
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norms
    return emb.astype("float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # normalize vektörlerle cosine ~ inner product
    index.add(vectors)
    return index


def retrieve_context(
    embedder: SentenceTransformer,
    question: str,
    chunks: List[str],
    index: faiss.IndexFlatIP,
    k: int = 4,
) -> Tuple[str, np.ndarray, List[str]]:
    """
    Soru embedding'i ile en benzer k chunk'ı bulur.
    Döndürdükleri:
      - context_text: Seçilen chunk'ların birleşimi
      - scores: FAISS skorları
      - selected_chunks: Tek tek chunk listesi
    """
    q_emb = embed_texts(embedder, [question])  # (1, dim)
    scores, indices = index.search(q_emb, k)
    selected_chunks = [chunks[i] for i in indices[0]]
    context_text = "\n\n---\n\n".join(selected_chunks)
    return context_text, scores[0], selected_chunks


# -------------------------
# ANSWER GENERATION (LOCAL)
# -------------------------
def split_sentences(text: str) -> List[str]:
    """
    Çok basit cümle bölücü; ödev için yeterli.
    İstersen noktaya göre böler, kısa parçaları filtreler.
    """
    raw = text.replace("\n", " ")
    parts = raw.split(".")
    sentences = [s.strip() for s in parts if len(s.strip()) > 20]
    return sentences


def simple_answer_from_context(
    embedder: SentenceTransformer,
    question: str,
    selected_chunks: List[str],
    top_n_sentences: int = 4,
) -> Tuple[str, List[str]]:
    """
    - Seçilen chunk'lardaki cümleleri çıkarır
    - Her cümleyi ve soruyu embed eder
    - Soruyla en benzer top-N cümleyi seçer
    - Bunları "cevap" olarak döner.
    """
    full_text = "\n ".join(selected_chunks)
    sentences = split_sentences(full_text)

    if not sentences:
        return "Belgede bu soruyla doğrudan ilgili bir cümle bulamadım.", []

    # Soru + cümleler için embedding
    q_emb = embed_texts(embedder, [question])  # (1, dim)
    sent_embs = embed_texts(embedder, sentences)  # (N, dim)

    # Cosine similarity (iç çarpım, vektörler normalize)
    sims = np.dot(q_emb, sent_embs.T)[0]  # (N,)

    # En benzer top-N cümleyi seç
    top_idx = np.argsort(-sims)[:top_n_sentences]
    top_sentences = [sentences[i].strip() for i in top_idx]

    answer_text = (
        f"Sorun: {question}\n\n"
        "Belgeden otomatik olarak seçilen en ilgili cümleler:\n\n"
        + "\n\n".join(f"- {s}" for s in top_sentences)
    )

    return answer_text, top_sentences


# -------------------------
# STREAMLIT UI (USER INTERFACE LAYER)
# -------------------------
def main():
    st.set_page_config(page_title="DocuChat (Free RAG)", layout="wide")

    # Sidebar: Architecture overview
    with st.sidebar:
        st.title("Architecture")
        st.markdown("**User Interface Layer**")
        st.caption("Streamlit web app: PDF yükleme + soru input")
        st.markdown("**Document Processing Layer**")
        st.caption("PyPDF2 ile parse, cleaning, chunking")
        st.markdown("**Vector Search Layer**")
        st.caption("SentenceTransformer embeddings + FAISS index")
        st.markdown("**Answering Layer (Local)**")
        st.caption("Soru–cümle benzerliği ile extractive özet")

        st.markdown("---")
        st.markdown("**Model:** `all-MiniLM-L6-v2`")
        st.markdown("**Index:** FAISS (Inner Product)")

    st.title("DocuChat – PDF ile Sohbet (Ücretsiz RAG Sürümü)")

    st.markdown(
        """
        Bu sürümde **OpenAI API kullanılmıyor**.  
        Tüm işlem local modellerle ve FAISS ile yapılıyor:
        1. PDF'ten metin çıkarma ve temizleme  
        2. Metni anlamlı parçalara (chunk) ayırma  
        3. Her chunk için embedding üretme  
        4. Soruya en benzeyen chunk'lardaki cümleleri seçip cevaplama  
        """
    )

    if "history" not in st.session_state:
        st.session_state.history = []  # (question, answer, context)

    uploaded_file = st.file_uploader("PDF dosyası yükle", type=["pdf"])

    if uploaded_file is None:
        st.info("Başlamak için bir PDF yükleyin.")
        return

    embedder = load_embedder()

    # PDF ilk kez yüklendiğinde işleme
    if "chunks" not in st.session_state:
        with st.spinner("PDF okunuyor ve işleniyor..."):
            pdf_bytes = io.BytesIO(uploaded_file.read())
            raw_text = parse_pdf(pdf_bytes)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned, max_tokens=800, overlap=100)

            # Çok büyük dosyalarda maliyeti azaltmak için chunk sayısını sınırla
            MAX_CHUNKS = 80
            if len(chunks) > MAX_CHUNKS:
                chunks = chunks[:MAX_CHUNKS]

            vectors = embed_texts(embedder, chunks)
            index = build_faiss_index(vectors)

            st.session_state.chunks = chunks
            st.session_state.index = index

        st.success(f"Belge işlendi. Chunk sayısı: {len(st.session_state.chunks)}")

    # Layout: Soru solda, sonuçlar aşağıda
    question = st.text_input("Belge hakkında bir soru sor:")

    if question:
        with st.spinner("İlgili context ve cümleler aranıyor..."):
            context, scores, selected_chunks = retrieve_context(
                embedder,
                question,
                st.session_state.chunks,
                st.session_state.index,
                k=4,
            )
            answer, top_sentences = simple_answer_from_context(
                embedder,
                question,
                selected_chunks,
            )

        st.session_state.history.append((question, answer, context, scores, top_sentences))

    # Geçmişi göster
    if st.session_state.history:
        st.subheader("Soru–Cevap Geçmişi")
        for q, a, _, _, _ in reversed(st.session_state.history):
            st.markdown(f"**Soru:** {q}")
            st.markdown("**Cevap:**")
            st.markdown(a)
            st.markdown("---")

    # Son sorunun context'i ve skorları
    if st.session_state.history:
        last_q, last_a, last_ctx, last_scores, last_top_sents = st.session_state.history[-1]

        with st.expander("Son soru için kullanılan context ve skorlar"):
            st.markdown("**Top-k chunk skorları (FAISS inner product):**")
            st.write(last_scores)
            st.markdown("---")
            st.markdown("**Context (birleştirilmiş chunk'lar):**")
            st.write(last_ctx)

        with st.expander("Son soruda seçilen en ilgili cümleler"):
            for s in last_top_sents:
                st.markdown(f"- {s}")


if __name__ == "__main__":
    main()
