# URL Phishing Detection Dataset  
*A curated dataset for machine learning and deep learning–based phishing URL detection*

---

## Overview

The **URL Phishing Detection Dataset** is a cleaned, structured, and research-ready dataset designed to support the development of machine learning, deep learning, and hybrid cybersecurity models. Phishing URLs remain one of the most widely used attack vectors for credential theft, financial scams, and social engineering attacks. Reliable predictive models require **large, diverse, and balanced datasets**, which are often difficult to obtain.

This repository provides a **high-quality labeled dataset** containing verified legitimate URLs and real phishing samples, enabling researchers and practitioners to build effective phishing detection systems.

---

## Dataset Contents

This repository includes the following CSV files:

| File Name              | Description |
|------------------------|-------------|
| **balanced_urls.csv**  | A merged, class-balanced dataset containing legitimate and phishing URLs in equal proportions. Ideal for unbiased ML/DL model training. |
| **legitimate_urls.csv** | A collection of verified safe URLs sourced from reputable domains and manually filtered to ensure accuracy. |
| **phishing_urls.csv**  | A collection of real phishing URLs sourced from threat-intelligence feeds, cybersecurity reports, and OSINT repositories. |

Each file contains labeled samples ready for preprocessing, feature extraction, and model training.

---

## Why This Dataset Matters

Building robust phishing-detection systems is challenging due to:

- Rapid evolution of phishing techniques  
- Limited access to clean, research-grade datasets  
- Severe class imbalance in real-world data  
- Lack of open, reproducible datasets accompanying published research  

This dataset was created specifically to address these issues:

###  Clean, validated samples  
Malformed URLs, duplicates, and noise were removed.

###  Balanced distributions  
`balanced_urls.csv` eliminates model bias by providing equal representation of classes.

###  Real-world attack samples  
Phishing URLs were collected from active feeds, making them representative of evolving threats.

###  Research-ready structure  
Ideal for ML (TF-IDF, n-grams) and DL (BERT, CNN, LSTM, Transformers) pipelines.

---

##  Visual Dataset Description

### Class Distribution in Balanced Dataset  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/bfd4c542-6e5d-4d15-a695-5575d9214f0c" />



### File Size Comparison  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/3a67c2ba-5db7-4c51-b0eb-3dcf025600ec" />


### Overall Class Proportion  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/97588fd0-6c89-4e83-9794-19c4bef3a708" />

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

---

##  Preprocessing Script

A preprocessing script is provided to:

- Normalize labels  
- Shuffle URLs  
- Split into train/validation/test sets  
- Re-generate balanced datasets  

Save the file as **`preprocess_dataset.py`**:

```python
#!/usr/bin/env python3
"""
Preprocessing utilities for the URL_Phishing_Detection_Dataset.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "."
OUTPUT_DIR = "./processed"


def load_datasets():
    balanced = pd.read_csv(os.path.join(DATA_DIR, "balanced_urls.csv"))
    legit = pd.read_csv(os.path.join(DATA_DIR, "legitimate_urls.csv"))
    phish = pd.read_csv(os.path.join(DATA_DIR, "phishing_urls.csv"))
    return balanced, legit, phish


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].str.lower().map({"legitimate": 0, "phishing": 1})
    df = df.dropna(subset=["url", "label"])
    df = df[["url", "label"]]
    return df


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

    train_b, val_b, test_b = train_val_test_split(balanced)
    train_b.to_csv(os.path.join(OUTPUT_DIR, "balanced_train.csv"), index=False)
    val_b.to_csv(os.path.join(OUTPUT_DIR, "balanced_val.csv"), index=False)
    test_b.to_csv(os.path.join(OUTPUT_DIR, "balanced_test.csv"), index=False)

    balanced_new = create_balanced_dataset(legit, phish)
    train_n, val_n, test_n = train_val_test_split(balanced_new)
    train_n.to_csv(os.path.join(OUTPUT_DIR, "new_balanced_train.csv"), index=False)
    val_n.to_csv(os.path.join(OUTPUT_DIR, "new_balanced_val.csv"), index=False)
    test_n.to_csv(os.path.join(OUTPUT_DIR, "new_balanced_test.csv"), index=False)

    print("Preprocessing complete. Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
```
 Run the above code using: **`python preprocess_dataset.py`**
