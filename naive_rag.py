import os
from dotenv import load_dotenv
import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# custom embedding function using SentenceTransformer
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

# create persistent Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

# 5. create or load collection with custom embedding function
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

# Listing out available models
# for m in genai.list_models():
#     print(m.name, m.supported_generation_methods)


# # Example usage of Gemini
# prompt = """You are a helpful assistant.

# What is human life expectancy in the United States?
# """

# # Generate response
# response = model.generate_content(prompt)
# print("Gemini says:", response.text)



# load a list of .txt documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text_into_chunks(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

print(f"Split documents into {len(chunked_documents)} chunks")


print("==== Upserting all chunked documents into ChromaDB ====")
collection.upsert(
    documents=[doc["text"] for doc in chunked_documents],
    ids=[doc["id"] for doc in chunked_documents],
    metadatas=[{"source": doc["id"]} for doc in chunked_documents],
)

# Query documents
def query_documents(question, n_results=2):
    # Chroma automatically embeds question using our custom embedding function we created
    results = collection.query(query_texts=[question], n_results=n_results)

    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")
    return relevant_chunks

# Generate a response
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    
    print("==== Sending prompt to Gemini ====")
    response = gemini_model.generate_content(prompt)
    
    return response.text

# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)

# Downfalls of this naive RAG
# 1. Limited contextual understanding -> RAG focuses on keyword matching or basic semantic similarity. It won't however contextualize what the query is actually asking though.
# 2. Incosistent relevance and quality of retrieved documents -> RAG retrieves documents based on keyword matching or semantic similarity, but those documents may not be relevant or high-quality.
# 3. Poor integration between retrieval and generation -> RAG retrieves documents first and then generates a response. This two-step process can lead to inconsistencies or generic responses.
# 4. Inefficent handling of large-scale data -> scaling isssues, takes too long to find relevant documents or miss info due to bad indexing.
# 5. Lack of robustness and adaptability -> RAG is not robust to noisy or incomplete data or queries. It may not handle new or complex queries well.


# Pre-retrieval to improve indexing structure and user's query's query and data details and aligning contexts better
# Post-retrieval to combine the pre-retrieval with query, like re-ranking etc.

# Query expansion -> expanding the query to include synonyms or related terms to improve retrieval
# -> We first pass through LLM and we generate our answer which then we augment with information from vector DB and re-query LLM to get a better answer.
# this can help with research because it can help find more relevant scientific papers or articles that may not contain the exact keywords used in the query.