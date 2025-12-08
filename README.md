# HSRW RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses documents from Rhine-Waal University of Applied Sciences (HSRW) to answer questions.

## Setup

**Prerequisite:** Ensure you have Python 3.11.9 installed and available.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download and run Ollama:**
    -   Download from [ollama.ai](https://ollama.ai/)
    -   Pull the required model:
        ```bash
        ollama pull all-minilm
        ```
5.  **Add PDF documents:**
    -   Place your PDF files into the `data/` directory.

## Usage

1.  **Ingest the documents:**
    -   This will create a local vector database from your PDFs.
    ```bash
    python ingest.py
    ```
2.  **(TODO) Run the chatbot:**
    ```bash
    python app.py
    ```
