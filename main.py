# main.py
from src.ingest import extract_text_by_page, build_faiss_index
from src.qa_bot import retrieve_answer

def main():
    pdf_path = r"C:\qa_chatbot\data\draft-oasis-e1-manual-04-28-2024.pdf"

    print("Extracting text from PDF...")
    pages_text = extract_text_by_page(pdf_path)

    print("Building FAISS index...")
    index, embeddings, chunks, embedder = build_faiss_index(pages_text)

    print("\nQ&A Bot Ready! Type 'exit' to quit.\n")

    while True:
        question = input("Q: ")
        if question.lower() == "exit":
            print("Exiting Q&A bot. Goodbye!")
            break

        # Retrieve answer using RAG mode (can also switch to "extractive")
        ans = retrieve_answer(
            question,
            index,
            embeddings,
            chunks,
            embedder,
            top_k=5,
            mode="rag"  # or "extractive"
        )

        print(f"\nA: {ans['answer']}")
        print(f"(Confidence: {ans['confidence']:.2f}, Pages: {ans['pages']})\n")

if __name__ == "__main__":
    main()
