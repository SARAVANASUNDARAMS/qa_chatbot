import streamlit as st
from pathlib import Path

from src.ingest import extract_text_by_page, build_faiss_index
from src.qa_bot import retrieve_answer

st.set_page_config(
    page_title="QA Chatbot",
    page_icon="🤖"
)

st.title("🤖 QA Chatbot")
st.write("Ask questions from the PDF manual.")

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "draft-oasis-e1-manual-04-28-2024.pdf"


@st.cache_resource
def load_qa_system():
    pages_text = extract_text_by_page(str(PDF_PATH))
    return build_faiss_index(pages_text)


index, embeddings, chunks, embedder = load_qa_system()

st.success("Q&A Bot Ready!")

question = st.text_input(
    "Enter your question:",
    placeholder="Ask something about the manual..."
)

if st.button("Ask"):
    if question.strip():
        ans = retrieve_answer(
            question,
            index,
            embeddings,
            chunks,
            embedder,
            top_k=5,
            mode="rag"
        )

        st.subheader("Answer")
        st.write(ans["answer"])

        st.caption(
            f"Confidence: {ans['confidence']:.2f} | "
            f"Pages: {ans['pages']}"
        )
    else:
        st.warning("Please enter a question.")
