# CMPS_Assignment5

## Overview

This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** system for video content. It allows users to ask questions about video content and retrieve relevant segments using various retrieval methods. The system supports both semantic and lexical retrieval techniques.

## Features

- **Video Processing**:
  - Downloads YouTube videos.
  - Transcribes audio using Whisper.
  - Extracts video frames at regular intervals.

- **Embeddings Generation**:
  - Generates text embeddings using Sentence Transformers.
  - Generates image embeddings using CLIP.

- **Retrieval Methods**:
  - **Semantic Retrieval**:
    - FAISS (Flat L2 Index).
    - PostgreSQL vector search (IVFFLAT and HNSW indexes).
  - **Lexical Retrieval**:
    - TF-IDF.
    - BM25.

- **Evaluation**:
  - Evaluates retrieval methods based on accuracy, rejection quality, and latency.

- **Streamlit Web App**:
  - Interactive interface for asking questions and retrieving video segments.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/alineFhassan/CMPS_Assignment5.git
   cd CMPS_Assignment5

2. **Install Dependencies: Install the required Python packages using pip:**
```pip install -r requirements.txt```

3. **Run the Scripts:**
- Prepare Data:
```python prepare_data.py```
- Generate Embeddings:
```python embeddings.py```
- Build Retrieval Models:
```python retrieval.py```
- Evaluate Retrieval Methods:
```python evaluation.py```
- Launch the Web App: Run the Streamlit app:
```streamlit run app.py```

## Project Structure
```
.
├── app.py
│   └── Streamlit web app for user interaction
├── embeddings.py
│   └── Generate text and image embeddings
├── evaluation.py
│   └── Evaluate retrieval methods (accuracy, rejection)
├── prepare_data.py
│   └── Download video, transcribe audio, extract frames
├── retrieval.py
│   └── Build retrieval models and indexes
├── retrieval_functions.py
│   └── Query functions for different retrieval methods
├── evaluation_results.csv
│   └── Evaluation results for retrieval methods
├── gold_standard_test_set.xlsx
│   └── Gold standard test set for evaluation
├── requirements.txt
│   └── Python dependencies
└── README.md
    └── Project documentation (this file)
```


## Usage
Streamlit Web App
- Open the Streamlit app in your browser.
- Select a retrieval method from the sidebar.
- Enter a question about the video content.
- View the retrieved video segments and their timestamps.

## Evaluation
The evaluation script computes:
- Accuracy on Answerable Questions: Measures how often the correct segment is retrieved.
- Rejection Quality on Unanswerable Questions: Measures the system's ability to reject unanswerable questions.
- Average Latency: Measures the time taken to retrieve results.
- Results are saved in evaluation_results.csv.

## Retrieval Methods Comparison

| Retrieval Method   | Accuracy on Answerable Questions | Rejection Quality on Unanswerable Questions | Average Latency (seconds) |
|---------------------|----------------------------------|---------------------------------------------|---------------------------|
| FAISS              | 0.7                              | 0.0                                         | 0.013                     |
| pgvector-IVFFLAT   | 0.6                              | 0.0                                         | 0.079                     |
| pgvector-HNSW      | 0.6                              | 0.0                                         | 0.07                      |
| TF-IDF             | 0.7                              | 0.0                                         | 0.001                     |
| BM25               | 0.4                              | 0.0                                         | 0.002                     |

