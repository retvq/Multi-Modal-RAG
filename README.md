# Multi-Modal RAG QA System

A production-ready Multi-Modal Retrieval-Augmented Generation (RAG) system capable of processing and answering questions from documents containing text, tables, and figures.

![20260111-1657-49 1187260-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/b953c176-ab0c-4a70-b223-a8a4fb70b67f)


## Features

- **Multi-Modal Ingestion**: Extracts and processes text, tables, and figures from PDFs.
- **Hybrid Retrieval**: Semantic search powered by SentenceTransformers and ChromaDB.
- **LLM Integration**: Generates answers using Google Gemini 2.0 Flash text-to-text and multi-modal capabilities.
- **Interactive UI**: Streamlit-based interface for QA, interacting with documents, and viewing retrieved context (citations).
- **Evaluation Dashboard**: Built-in dashboard to view system metrics and query performance.
- **Automated Fallbacks**: gracefully falls back to rule-based methods if LLM quotas are exceeded.

## Project Structure

```
├── src/                # Source code for ingestion, retrieval, and generation
├── tests/              # Unit and integration tests
├── scripts/            # Utility and maintenance scripts
├── data/               # Input documents (PDFs)
├── outputs/            # Generated vectors, logs, and metadata
├── app.py              # Main Streamlit Application
├── dashboard.py        # Evaluation Dashboard
├── demo.py             # CLI/Scriptable Demo
└── requirements.txt    # Project dependencies
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    ```

## Usage

### Run the Web Interface
```bash
streamlit run app.py
```

### Run the Evaluation Dashboard
```bash
streamlit run dashboard.py
```

### Run CLI Demo
```bash
python demo.py
```

## Tech Stack

- **LLM**: Google Gemini
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF, Tabula, PDF2Image




