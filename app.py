import io
import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------------
# ÜCRETSİZ SÜRÜM (OpenAI YOK)
# - Embedding: all-MiniLM-L6-v2 (sentence-transformers)
# - Retrieval: FAISS
# - Cevap: İlgili parçaları gösteren basit özet
# -----------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# -------------------------
# DOCUMENT PROCESSING
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


def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100):
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
# EMBEDDINGS + FAISS INDEX (LOCAL)
# -------------------------
def embed_texts(embedder, texts):
    # sentence-transformers, numpy array döner
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Cosine similarity için normalize edelim
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norms
    return emb.astype("float32")


def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine ~ Inner Product (normlanmış vektörler)
    index.add(vectors)
    return index


def retrieve_context(embedder, question, chunks, index, k: int = 4):
    q_emb = embed_texts(embedder, [question])  # shape (1, dim)
    scores, indices = index.search(q_emb, k)
    selected = [chunks[i] for i in indices[0]]
    context = "\n\n---\n\n".join(selected)
    return context, scores[0]


# -------------------------
# "CEVAP" OLUŞTURMA (LLM YOK)
# -------------------------
def simple_answer_from_context(question: str, context: str) -> str:
    """
    Gerçek LLM kullanmak yerine:
    - En ilgili context parçalarını alıyoruz
    - İçinden ilk birkaç cümleyi kullanıcıya "cevap" gibi gösteriyoruz
    Bu tamamen ÜCRETSİZ, ama yine de RAG mantığını göstermeye yetiyor.
    """
    if not context.strip():
        return "Belgede bu soruyla ilgili bir bilgi bulamadım."

    # En basit haliyle: ilk 3–4 cümleyi al
    sentences = context.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:4])

    return (
        f"Sorun: {question}\n\n"
        f"Belgeden bulunan ilgili kısımlar:\n\n{summary.strip()}..."
    )


# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="DocuChat (Free Version)", layout="wide")
    st.title("DocuChat – PDF ile Sohbet (Ücretsiz Sürüm)")

    st.markdown(
        "- Bu sürümde **OpenAI API kullanılmıyor**, tamamen ücretsiz.\n"
        "- Embedding için `all-MiniLM-L6-v2`, arama için FAISS kullanıyoruz.\n"
        "- Cevap olarak, belgeden en ilgili paragrafları gösteriyoruz."
    )

    if "history" not in st.session_state:
        st.session_state.history = []  # (soru, cevap, context)

    uploaded_file = st.file_uploader("PDF dosyası yükle", type=["pdf"])

    if uploaded_file is None:
        st.info("Başlamak için bir PDF yükleyin.")
        return

    # Modeli yükle
    embedder = load_embedder()

    # PDF ilk kez yüklendiğinde işleme
    if "chunks" not in st.session_state:
        with st.spinner("PDF okunuyor ve işleniyor..."):
            pdf_bytes = io.BytesIO(uploaded_file.read())
            raw_text = parse_pdf(pdf_bytes)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned, max_tokens=800, overlap=100)

            # Çok büyük dosyalarda masrafı azaltmak için chunk sayısını sınırlayalım
            MAX_CHUNKS = 80
            if len(chunks) > MAX_CHUNKS:
                chunks = chunks[:MAX_CHUNKS]

            vectors = embed_texts(embedder, chunks)
            index = build_faiss_index(vectors)

            st.session_state.chunks = chunks
            st.session_state.index = index

        st.success(f"Belge işlendi. Chunk sayısı: {len(st.session_state.chunks)}")

    question = st.text_input("Belge hakkında bir soru sor:")

    if question:
        with st.spinner("İlgili context aranıyor..."):
            context, scores = retrieve_context(
                embedder,
                question,
                st.session_state.chunks,
                st.session_state.index,
                k=4,
            )
            answer = simple_answer_from_context(question, context)

        st.session_state.history.append((question, answer, context))

    # Geçmişi göster
    if st.session_state.history:
        st.subheader("Soru–Cevap Geçmişi")
        for q, a, _ in reversed(st.session_state.history):
            st.markdown(f"**Soru:** {q}")
            st.markdown(f"**Cevap:**")
            st.markdown(a)
            st.markdown("---")

    if st.session_state.history and st.checkbox("Son soru için kullanılan context'i göster"):
        _, _, ctx = st.session_state.history[-1]
        st.write(ctx)


if __name__ == "__main__":
    main()

