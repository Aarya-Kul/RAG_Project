from helper_utils import word_wrap
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import numpy as np

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

def generate_gemini_answer(query, context):
    prompt = f"""
            You are a knowledgeable financial research assistant.
            Your users are asking about an annual report.

            Based on the following context:

            {context}

            Answer the question: '{query}'
            Use three to four concise sentences. If unsure, say you are unsure.
            """
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_gemini_answer_no_rag(query):
    prompt = f"""
            You are a knowledgeable financial research assistant.
            Your users are asking about an annual report.

            Answer the question: '{query}'
            Use three to four concise sentences. If unsure, say you are unsure.
            """
    response = model.generate_content(prompt)
    return response.text.strip()

original_query = (
    "What were the most important factors that contributed to increases in revenue for Microsoft in 2023?"
)

# ========== 1. Original Query with No RAG ==========
print("\nGemini Response with Original Query (No RAG):\n")
final_answer = generate_gemini_answer_no_rag(original_query)
print(word_wrap(final_answer))


# load PDF and extract text
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
full_text = "\n\n".join(pdf_texts)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
char_chunks = character_splitter.split_text(full_text)

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_chunks = []
for text in char_chunks:
    token_chunks += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    name="microsoft-collect",
    embedding_function=embedding_function
)

ids = [str(i) for i in range(len(token_chunks))]
chroma_collection.add(ids=ids, documents=token_chunks)


# ========== 2. Original Query with RAG ==========
# GETS TOP 10 CHUNKS THAT ARE RELEVANT
results_rag = chroma_collection.query(
    query_texts=[original_query], n_results=10, include=["documents"]
)
retrieved_documents = results_rag["documents"][0]
context_rag = "\n\n".join(retrieved_documents)
print("\nGemini Response with Original Query + RAG:\n")
final_answer = generate_gemini_answer(original_query, context_rag)
print(word_wrap(final_answer))


# Unlike SentenceTransformer (which embeds query/doc separately), CrossEncoder jointly encodes the query and document for more accurate ranking.

# SOLVES ISSUES WITH:
# Shallow matches
# Irrelevant chunks
# Keyword-only similarity
# Generic LLM answers

# ========== 3. Original Query with CrossEncoder RAG ==========
# Cross-Encoder Reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[original_query, doc] for doc in retrieved_documents]
# You sort those matches with a deep neural network that actually reads and understands both the query and the documents.
scores = cross_encoder.predict(pairs)
print("Cross-Encoder Scores:")
for score in scores:
    print(score)
print("\n Reranked Order (Top = Best):")
for o in np.argsort(scores)[::-1]:
    print(o + 1)

reranked_docs_ce = [retrieved_documents[i] for i in np.argsort(scores)[::-1]]
context_ce_rag = "\n\n".join(reranked_docs_ce[:5])
print("\n\nGemini Response with Original Query + CrossEncoder RAG:\n")
final_answer = generate_gemini_answer(original_query, context_ce_rag)
print(word_wrap(final_answer))


# ========== 4. Expanded Query with RAG ==========
# Query Expansion
generated_queries = [
    "What were the major drivers of for Microsoft's revenue growth in 2023?",
    "Were there any new product launches in 2023 that contributed to the increase in Microsoft's revenue?",
    "Did any changes in pricing or promotions in 2023 impact the revenue growth for Microsoft?",
    "What were the key 2023 market trends that facilitated the increase in Microsoft's revenue?",
    "Did any acquisitions or partnerships contribute to Microsoft's revenue growth in 2023?",
]

queries = [original_query] + generated_queries

results = chroma_collection.query(
    query_texts=queries,
    n_results=10,
    include=["documents", "embeddings"]
)

# Remove duplicates and rerank results
retrieved_documents = results["documents"]
unique_documents = list(set(doc for group in retrieved_documents for doc in group))
context_expanded_rag = "\n\n".join(unique_documents[:10])
print("\n\nGemini Response with Expanded Query + RAG:\n")
final_answer = generate_gemini_answer(original_query, context_expanded_rag)
print(word_wrap(final_answer))


# ========== 5. Expanded Query with CrossEncoder RAG ==========
pairs = [[original_query, doc] for doc in unique_documents]
scores = cross_encoder.predict(pairs)
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]
context_expanded_ce_rag = "\n\n".join(top_documents)
print("\n\nGemini Response with Expanded Query + CrossEncoder RAG:\n")
final_answer = generate_gemini_answer(original_query, context_expanded_ce_rag)
print(word_wrap(final_answer))


# query = "What has been the investment in research and development?"
# results = chroma_collection.query(
#     query_texts=[query], n_results=10, include=["documents", "embeddings"]
# )

# retrieved_documents = results["documents"][0]

# print("\n Top Retrieved Chunks:\n")
# for doc in retrieved_documents:
#     print(word_wrap(doc))
#     print("")









# final_answer = generate_gemini_answer(original_query, context)

# print("Original Query:\n", original_query)
# print("\n Final Answer from Gemini:\n")
# print(word_wrap(final_answer))
