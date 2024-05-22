# Internal Document RAG System

## Overview

This project is an implementation of a Retrieval-Augmented Generation (RAG) system designed to process and retrieve relevant information from internal company documents, specifically earnings reports. The system leverages OpenAI's GPT-4o Vision model for semantic chunking and embedding generation, and ZillizDB (Milvus) for vector storage and retrieval.

## Features

- **Embedder:** `text-embedding-3-large`
- **Model:** GPT-4o Vision
- **Chunking Method:** Semantic Chunking by paragraph/section/table
- **Distance Metric:** Euclidean Distance (L2)
- **Index Type:** IVF_FLAT
- **Metadata Filters:** Company name, Document type, Date

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- pip (Python package installer)
- Git

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/internal-document-rag-system.git
    cd internal-document-rag-system
    ```

2. **Set up virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required Python packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1. **API Keys and ZillizDB Credentials:**

    Update the `api_key` and `zilliz_credentials` variables in the main script with your OpenAI and ZillizDB credentials.

2. **Folder Path:**

    Set the `folder_path` variable to the directory containing your PDF documents.

### Running the System

1. **Data Ingestion Pipeline:**

    The data ingestion pipeline processes PDF documents, extracts text using GPT-4o Vision, performs semantic chunking, generates embeddings, and stores them in ZillizDB.

    ```python
    from data_ingestion_pipeline import run_data_ingestion_pipeline
    run_data_ingestion_pipeline()
    ```

2. **Query Pipeline:**

    The query pipeline processes user queries, retrieves relevant information from ZillizDB, and returns an enhanced prompt with contextual information.

    ```python
    from query_pipeline import run_query_pipeline
    run_query_pipeline(query
