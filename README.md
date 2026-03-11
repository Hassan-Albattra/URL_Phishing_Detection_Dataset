# URL Phishing Detection Dataset  
*A curated dataset for machine learning and deep learning–based phishing URL detection*
---
## Overview
To ensure robust model training and evaluation, we compiled a comprehensive dataset of **340,000 URLs** from multiple authoritative sources. The dataset is evenly divided into:
- **170,000 phishing URLs**
- **170,000 legitimate URLs**
GuardedRAG is a secure Retrieval-Augmented Generation (RAG) system designed to answer questions based on uploaded documents while enforcing strict safety and educational guardrails. The system combines document retrieval, large language models, and content moderation to ensure that responses remain accurate, context-grounded, and safe.

This balanced composition prevents class bias and ensures fair and reliable evaluation for machine learning (ML) and deep learning (DL) models.
Users can upload PDF or text documents, which are automatically processed, chunked, embedded, and stored in a vector database. When a question is submitted, the system retrieves the most relevant document passages and generates an answer using those passages as context.

The dataset was collected **between January 2023 and December 2023**, ensuring that it reflects modern phishing techniques and evolving cyber-attack patterns.
Safety checks are applied both before and after answer generation to prevent harmful or non-educational queries.

---

##  Dataset Files
### **1. `balanced_urls.csv`**
This file contains the **fully balanced dataset**, consisting of **170,000 phishing URLs** and **170,000 legitimate URLs**.  
It is ideal for ML/DL experimentation where **class balance** is critical for preventing model bias.
### **2. `legitimate_urls.csv`**
Legitimate URLs were sourced from multiple high-quality, real-world web data repositories:
- **Alexa Top Sites** – High-traffic, globally ranked websites.
- **Common Crawl** – A large-scale, continuously updated web corpus.
# Key Features

These sources ensure:
- Diversity across industries and domains  
- Updated, real-world browsing behavior  
- Reduction of accidental inclusion of unsafe or inactive URLs  
### **3. `phishing_urls.csv`**
Phishing URLs were collected from authoritative, frequently updated cybersecurity datasets:
- **PhishTank** – Community-verified phishing submissions, manually validated by experts.
- **OpenPhish** – A commercial automated feed of high-confidence phishing URLs discovered using proprietary algorithms.
These sources are widely used in cybersecurity research and ensure a realistic representation of **active and modern phishing threats**.
- Document-based Question Answering  
- Safety Guardrails for input and output moderation  
- Educational Scope Filtering  
- Semantic Retrieval using Vector Search  
- Fast Context-Grounded Answer Generation  
- Interactive Web Interface

---

## Why This Dataset Matters
Building robust phishing-detection systems is challenging due to:
# System Architecture

- Rapid evolution of phishing techniques  
- Limited access to clean, research-grade datasets  
- Severe class imbalance in real-world data  
- Lack of open, reproducible datasets accompanying published research  
The GuardedRAG pipeline follows five stages:

Although the dataset sources differ from prior works, we ensure comparability by:
## 1. Scope Check
Ensures that the query is educational or academic in nature.

- Using standard evaluation metrics (Accuracy, Precision, Recall, F1-score)  
- Clearly documenting dataset size, class distribution, and sources  
- Aligning evaluation methodology with common practices in the field  
## 2. Input Moderation
Detects harmful or unsafe user queries before they reach the language model.

This dataset was created specifically to address these issues:
## 3. Context Retrieval
Retrieves the most relevant document chunks from the vector database.

###  Clean, validated samples  
Malformed URLs, duplicates, and noise were removed.
## 4. Answer Generation
Generates responses using the retrieved document context.

###  Balanced distributions  
`balanced_urls.csv` eliminates model bias by providing equal representation of classes.
###  Real-world attack samples  
Phishing URLs were collected from active feeds, making them representative of evolving threats.
###  Research-ready structure  
Ideal for ML (TF-IDF, n-grams) and DL (BERT, CNN, LSTM, Transformers) pipelines.
## 5. Output Moderation
Ensures the generated response is safe before returning it to the user.

---

##  Visual Dataset Description
### Class Distribution in Balanced Dataset  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/bfd4c542-6e5d-4d15-a695-5575d9214f0c" />
# Models Used

## Worker Model
**Gemini 2.5 Flash**

Responsible for generating answers based on the retrieved document context.

### File Size Comparison  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/3a67c2ba-5db7-4c51-b0eb-3dcf025600ec" />
## Guardrail Model
**OpenAI Moderation Model (omni-moderation-latest)**

Detects harmful or unsafe content in both user queries and model outputs.

### Overall Class Proportion  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/97588fd0-6c89-4e83-9794-19c4bef3a708" />
## Scope Classification Model
**Gemini 2.5 Flash**

---
##  Research Applications
This dataset can be used for:
- Phishing URL classification  
- NLP-based URL embeddings  
- Deep learning model benchmarking  
- Hybrid BERT + TF-IDF architectures  
- Risk analysis and cyber-attack prediction  
- Adversarial robustness evaluation  
- Feature engineering experiments  
It supports the research study:  
**“Risk Analysis and Cyber Attack Prediction Based on Machine Learning and Deep Learning: A Case Study on Phishing URL Detection.”**
Ensures that the system only answers educational or academic queries.

---

##  Preprocessing Script
A preprocessing script is provided to:
- Normalize labels  
- Shuffle URLs  
- Split into train/validation/test sets  
- Re-generate balanced datasets  
# Document Processing Pipeline

Save the file as **`preprocess_dataset.py`**:
## Text Extraction
Documents are parsed using **PyMuPDF** to extract raw text.

```python
#!/usr/bin/env python3
"""
Preprocessing utilities for the URL_Phishing_Detection_Dataset.
"""
## Chunking
Documents are split into smaller sections using **LangChain RecursiveCharacterTextSplitter**.

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
Configuration:

DATA_DIR = "."
OUTPUT_DIR = "./processed"
- Chunk size: 500 characters  
- Chunk overlap: 80 characters  

## Embedding
Each chunk is converted into a vector representation using:

def load_datasets():
    balanced = pd.read_csv(os.path.join(DATA_DIR, "balanced_urls.csv"))
    legit = pd.read_csv(os.path.join(DATA_DIR, "legitimate_urls.csv"))
    phish = pd.read_csv(os.path.join(DATA_DIR, "phishing_urls.csv"))
    return balanced, legit, phish
**Sentence Transformers — all-MiniLM-L6-v2**

## Vector Storage
Embeddings are stored in **ChromaDB**, enabling efficient semantic similarity search.

def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].str.lower().map({"legitimate": 0, "phishing": 1})
    df = df.dropna(subset=["url", "label"])
    df = df[["url", "label"]]
    return df
## Retrieval
For each query, the system retrieves the **Top-K most relevant chunks** from the vector database.

---

def create_balanced_dataset(legit: pd.DataFrame, phish: pd.DataFrame) -> pd.DataFrame:
    n = min(len(legit), len(phish))
    legit_bal = legit.sample(n, random_state=42)
    phish_bal = phish.sample(n, random_state=42)
    balanced = pd.concat([legit_bal, phish_bal], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return balanced
def train_val_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42, stratify=train_df["label"])
    return train_df, val_df, test_df
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    balanced_raw, legit_raw, phish_raw = load_datasets()
    balanced = normalise_labels(balanced_raw)
    legit = normalise_labels(legit_raw)
    phish = normalise_labels(phish_raw)
# Tech Stack

    train_b, val_b, test_b = train_val_test_split(balanced)
    train_b.to_csv(os.path.join(OUTPUT_DIR, "balanced_train.csv"), index=False)
    val_b.to_csv(os.path.join(OUTPUT_DIR, "balanced_val.csv"), index=False)
    test_b.to_csv(os.path.join(OUTPUT_DIR, "balanced_test.csv"), index=False)
- Gemini API — answer generation and scope classification  
- OpenAI Moderation API — safety guardrails  
- ChromaDB — vector database  
- Sentence Transformers — text embeddings  
- LangChain Text Splitter — document chunking  
- PyMuPDF — PDF text extraction  
- Gradio — interactive web interface  

    balanced_new = create_balanced_dataset(legit, phish)
    train_n, val_n, test_n = train_val_test_split(balanced_new)
    train_n.to_csv(os.path.join(OUTPUT_DIR, "new_balanced_train.csv"), index=False)
    val_n.to_csv(os.path.join(OUTPUT_DIR, "new_balanced_val.csv"), index=False)
    test_n.to_csv(os.path.join(OUTPUT_DIR, "new_balanced_test.csv"), index=False)
---

    print("Preprocessing complete. Files saved to:", OUTPUT_DIR)
# Running the Project

## 1. Install Dependencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
```
 Run the above code using: **`python preprocess_dataset.py`**
```bash
pip install gradio chromadb sentence-transformers pymupdf langchain-text-splitters openai google-genai
0 commit comments
Comments
0
 (0)
Comment
You're not receiving notifications from this thread.

