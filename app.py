import io
import os

import streamlit as st
import PyPDF2
import numpy as np
import faiss
from openai import OpenAI

# -------------------------
# API KEY (Streamlit Secrets veya Environment'tan)
# -------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY bulunamadı. Lütfen Streamlit Secrets'e ekle.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"


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
# EMBEDDINGS + FAISS INDEX
# -------------------------
def embed_texts(texts):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    vectors = [np.array(item.embedding, dtype="float32") for item in resp.data]
    return vectors


def build_faiss_index(chunks):
    vecs = embed_texts(chunks)
    mat = np.vstack(vecs)
    dim = mat.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(mat)
    return index


def retrieve_context(question, chunks, index, k: int = 4):
    q_vec = embed_texts([question])[0]
    q_vec = np.expand_dims(q_vec, axis=0)
    distances, indices = index.search(q_vec, k)
    selected = [chunks[i] for i in indices[0]]
    context = "\n\n---\n\n".join(selected)
    return context


# -------------------------
# LLM CALL
# -------------------------
def call_llm(context: str, question: str) -> str:
    system_message = (
        "You are a helpful assistant that answers questions about a PDF document. "
        "Use ONLY the information from the provided context excerpts. "
        "If the answer is not in the context, say you do not know."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="DocuChat", layout="wide")
    st.title("DocuChat – PDF ile Sohbet")

    if "history" not in st.session_state:
        st.session_state.history = []  # (question, answer, context)

    st.markdown(
        "1. PDF yükle\n"
        "2. Sistem belgeyi işler (parse + chunk + embedding + index)\n"
        "3. Sorunu sor ve cevabı al."
    )

    uploaded_file = st.file_uploader("PDF dosyası yükle", type=["pdf"])

    if uploaded_file is None:
        st.info("Başlamak için bir PDF yükle.")
        return

    # PDF ilk kez yüklendiyse işleme yap
    if "chunks" not in st.session_state:
        with st.spinner("PDF okunuyor ve işleniyor..."):
            pdf_bytes = io.BytesIO(uploaded_file.read())
            raw_text = parse_pdf(pdf_bytes)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned, max_tokens=800, overlap=100)
            index = build_faiss_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index

        st.success(f"Belge yüklendi. Chunk sayısı: {len(st.session_state.chunks)}")

    question = st.text_input("Belge hakkında bir soru sor:")

    if question:
        with st.spinner("Context aranıyor ve cevap üretiliyor..."):
            context = retrieve_context(
                question,
                st.session_state.chunks,
                st.session_state.index,
                k=4,
            )
            answer = call_llm(context, question)

        st.session_state.history.append((question, answer, context))

    # Geçmişi göster
    if st.session_state.history:
        st.subheader("Soru–Cevap Geçmişi")
        for q, a, _ in reversed(st.session_state.history):
            st.markdown(f"**Soru:** {q}")
            st.markdown(f"**Cevap:** {a}")
            st.markdown("---")

    # Son context'i göster
    if st.session_state.history and st.checkbox("Son kullanılan context'i göster"):
        _, _, ctx = st.session_state.history[-1]
        st.write(ctx)


if __name__ == "__main__":
    main()
