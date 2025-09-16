# qa_bot.py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

# Load extractive QA model (for short, precise answers)
extractive_model_name = "distilbert-base-uncased-distilled-squad"
extractive_tokenizer = AutoTokenizer.from_pretrained(extractive_model_name)
extractive_model = AutoModelForQuestionAnswering.from_pretrained(extractive_model_name)

# Load RAG / seq2seq model (for generative answers)
rag_model_name = "google/flan-t5-base"  # supports longer sequences than GPT2
rag_tokenizer = AutoTokenizer.from_pretrained(rag_model_name)
rag_model = AutoModelForSeq2SeqLM.from_pretrained(rag_model_name)


def retrieve_answer(question, faiss_index, embeddings, chunks, embedder, top_k=5, mode="extractive"):
    """
    Retrieve an answer from PDF using FAISS + QA model.
    
    Args:
        question (str): user query
        faiss_index: FAISS index
        embeddings: embeddings array
        chunks: list of dicts {"text": chunk_text, "page": page_number}
        embedder: sentence transformer model
        top_k (int): number of top chunks to retrieve
        mode (str): "extractive" or "rag"
    
    Returns:
        dict: {"answer": ..., "confidence": ..., "pages": [...]}
    """
    # 1. Embed the query
    query_emb = embedder.encode(question)

    # 2. Search top_k similar chunks
    D, I = faiss_index.search(query_emb.reshape(1, -1), top_k)

    retrieved_chunks = [chunks[i] for i in I[0]]
    context_text = " ".join([c["text"] for c in retrieved_chunks])
    pages = [c["page"] for c in retrieved_chunks]

    # Truncate context for models with limited max length
    if mode == "extractive":
        inputs = extractive_tokenizer(
            question,
            context_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        outputs = extractive_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer_ids = inputs["input_ids"][0][answer_start:answer_end]
        answer_text = extractive_tokenizer.decode(answer_ids, skip_special_tokens=True)
        confidence = torch.max(torch.softmax(outputs.start_logits, dim=1)) * \
                     torch.max(torch.softmax(outputs.end_logits, dim=1))
    else:  # mode == "rag"
        # Truncate context to max 1024 tokens for generative model
        inputs = rag_tokenizer(
            context_text,
            question,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        outputs = rag_model.generate(**inputs, max_new_tokens=150)
        answer_text = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
        confidence = 1.0  # generative answer does not have explicit confidence

    return {
        "answer": answer_text,
        "confidence": float(confidence),
        "pages": pages
    }
