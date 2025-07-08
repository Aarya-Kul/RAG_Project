# 🔬 Scientific Literature Analysis with Retrieval-Augmented Generation (RAG)

This project explores how different Retrieval-Augmented Generation (RAG) pipelines can improve scientific literature understanding by allowing models to access up-to-date external knowledge at inference time.

We evaluate and compare several RAG variations for tasks like entity extraction and question answering over scientific texts.

Here is the link to the research paper: [Scientific Literature Analysis with Retrieval-Augmented Generation](https://github.com/Aarya-Kul/RAG_Project/blob/master/Enhancing%20Scientific%20Literature%20Analysis%20with%20a%20RAG%20System.pdf) and [Presentation](https://github.com/Aarya-Kul/RAG_Project/blob/master/Final_Presentation.pptx) showing our results. 

---

## 🧠 Implemented RAG Approaches

### 1. Naive RAG (Implemented in `naive_rag.py`)
- **LLM**: Google Gemini (via `google.generativeai`)
- **Embedding**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB (local persistent store)
- **Pipeline**:
  - Load `.txt` documents
  - Chunk and embed them
  - Store in Chroma
  - At query time, retrieve top-k chunks and use them as context in Gemini prompt

**Pros**:
- Simple and lightweight
- Easy to extend
- Works well for basic QA tasks

**Cons**:
- No reranking or filtering
- May retrieve semantically weak chunks
- Susceptible to hallucinations

---

### 2. Semantic RAG (Proposed)
- **Retriever**: Dense retriever like `contriever-msmarco`
- **Reranker**: Cross-encoder reranks top-k results
- **Use case**: Higher precision and grounded answers

---

### 3. Hybrid RAG (BM25 + Dense)
- **First pass**: BM25 filters by lexical match
- **Second pass**: Dense retriever refines based on semantics
- **Goal**: Combine high recall with better context

---

### 4. Query Augmentation RAG (Proposed)
- **Technique**: Expand original query using LLM (synonyms, related concepts)
- **Improves**: Chunk matchability and retrieval diversity
- **Downside**: Increases latency and complexity

---

## 🧪 Evaluation Strategy

- Benchmarked on biomedical abstracts and QA tasks using:
  - F1 / precision / recall
  - Hallucination rate
  - Human eval of factuality and relevance

- Results showed naive RAG improved baseline LLM performance, but semantic/hybrid RAG provided more grounded answers.

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your Gemini API key to a `.env` file:

```
GOOGLE_API_KEY="your-api-key-here"
```

3. Place `.txt` documents into `news_articles/`

4. Run the naive RAG pipeline:

```bash
python naive_rag.py
```

---

## 📌 Future Work

- Implement reranking and hybrid pipelines
- Add query rewriting + expansion modules
- Expand to larger scientific corpora (PubMed, arXiv)
- Apply to domain-specific evaluations like SciRIFF and BioRED

---

**Authors**: Aarya Kulshrestha, Omkar Vodela, Xuteng Luo, Jin Huang  
**University of Michigan – April 2025**
