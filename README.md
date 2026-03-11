# GuardedRAG — Safe Retrieval-Augmented Generation System

GuardedRAG is a secure Retrieval-Augmented Generation (RAG) system designed to answer questions based on uploaded documents while enforcing strict safety and educational guardrails. The system combines document retrieval, large language models, and content moderation to ensure that responses remain accurate, context-grounded, and safe.

Users can upload PDF or text documents, which are automatically processed, chunked, embedded, and stored in a vector database. When a question is submitted, the system retrieves the most relevant document passages and generates an answer using those passages as context.

Safety checks are applied both before and after answer generation to prevent harmful or non-educational queries.

---

# Key Features

- Document-based Question Answering  
- Safety Guardrails for input and output moderation  
- Educational Scope Filtering  
- Semantic Retrieval using Vector Search  
- Fast Context-Grounded Answer Generation  
- Interactive Web Interface

---

# System Architecture

The GuardedRAG pipeline follows five stages:

## 1. Scope Check
Ensures that the query is educational or academic in nature.

## 2. Input Moderation
Detects harmful or unsafe user queries before they reach the language model.

## 3. Context Retrieval
Retrieves the most relevant document chunks from the vector database.

## 4. Answer Generation
Generates responses using the retrieved document context.

## 5. Output Moderation
Ensures the generated response is safe before returning it to the user.

---

# Models Used

## Worker Model
**Gemini 2.5 Flash**

Responsible for generating answers based on the retrieved document context.

## Guardrail Model
**OpenAI Moderation Model (omni-moderation-latest)**

Detects harmful or unsafe content in both user queries and model outputs.

## Scope Classification Model
**Gemini 2.5 Flash**

Ensures that the system only answers educational or academic queries.

---

# Document Processing Pipeline

## Text Extraction
Documents are parsed using **PyMuPDF** to extract raw text.

## Chunking
Documents are split into smaller sections using **LangChain RecursiveCharacterTextSplitter**.

Configuration:

- Chunk size: 500 characters  
- Chunk overlap: 80 characters  

## Embedding
Each chunk is converted into a vector representation using:

**Sentence Transformers — all-MiniLM-L6-v2**

## Vector Storage
Embeddings are stored in **ChromaDB**, enabling efficient semantic similarity search.

## Retrieval
For each query, the system retrieves the **Top-K most relevant chunks** from the vector database.

---

# Tech Stack

- Gemini API — answer generation and scope classification  
- OpenAI Moderation API — safety guardrails  
- ChromaDB — vector database  
- Sentence Transformers — text embeddings  
- LangChain Text Splitter — document chunking  
- PyMuPDF — PDF text extraction  
- Gradio — interactive web interface  

---

# Running the Project

## 1. Install Dependencies

```bash
pip install gradio chromadb sentence-transformers pymupdf langchain-text-splitters openai google-genai
