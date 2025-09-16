# Q\&A Bot for PDF Manuals

## **Overview**

This project is a **document-based Q\&A bot** that allows users to query PDF manuals and retrieve precise answers along with relevant context and page numbers.
It leverages **FAISS** for vector-based retrieval and a **RAG (Retrieval-Augmented Generation)** model for generative answers.

---

## **Features**

* Extracts text from multi-page PDF documents.
* Chunks text into semantic units for efficient retrieval.
* Converts queries into embeddings for semantic search.
* Retrieves top-k relevant text chunks using FAISS.
* Generates coherent answers using a RAG model (`flan-t5-base`).
* Returns answers with confidence score and page references.
* Supports interactive Q\&A via command-line interface.

---

## **Architecture**

### **Pipeline**

1. **User Query Input**

   * Accepts questions from the user.
2. **Query Embedding**

   * Converts query into a vector representation using `sentence-transformers`.
3. **FAISS Retrieval**

   * Finds top-k relevant chunks from the document embedding index.
4. **RAG Model (Generative Answer)**

   * Generates the final answer from retrieved chunks and query.
5. **Output**

   * Returns answer text, confidence, and source page numbers.

### **Flow Diagram**

```
User Query
   │
   ▼
[Query Embedding]
   │
   ▼
[FAISS Retrieval of Top-k Chunks]
   │
   ▼
[RAG Model: Context + Query → Generate Answer]
   │
   ▼
Answer + Confidence + Page References
```

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/qa_bot.git
cd qa_bot
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK resources:

```python
import nltk
nltk.download('punkt')
```

---

## **Usage**

Run the interactive Q\&A bot:

```bash
python main.py
```

* Type a question related to the PDF manual.
* The bot returns the answer with confidence and page numbers.
* Type `exit` to quit.

---

## **Models Used**

| Component         | Model                 | Purpose                                          |
| ----------------- | --------------------- | ------------------------------------------------ |
| Query Embedding   | `all-mpnet-base-v2`   | Converts queries into semantic embeddings        |
| Generative Answer | `google/flan-t5-base` | Generates coherent answers from retrieved chunks |

---

## **Project Structure**

```
qa_chatbot/
├── data/                   # PDF manuals
├── src/
│   ├── ingest.py           # Text extraction, chunking, FAISS index
│   └── qa_bot.py           # Retrieval + RAG-based answer generation
├── main.py                 # Entry point for interactive Q&A
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```


---

## **Future Improvements**

* Support for multiple PDFs simultaneously.
* Web-based interface for better usability.
* Integration of more advanced LLMs for higher-quality answers.
* Add caching mechanism for faster repeated queries.

---

I can also draft a **very concise “badges + quick start” version** for GitHub if you want it to look **professional for recruiters**.

Do you want me to create that version too?
