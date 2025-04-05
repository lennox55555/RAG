# Mary Ferrell Foundation RAG Application

A Retrieval-Augmented Generation (RAG) application built for the Mary Ferrell Foundation. This system enables semantic search over historical documents related to the JFK assassination, civil rights, and other significant historical events.

## Features

- **Semantic Search**: Query historical documents using natural language
- **Document Management**: Upload and process PDF, CSV, and JSON files
- **Retrieval-Only Mode**: Get direct access to relevant source documents
- **Advanced Document Processing**: OCR for PDFs, chunking for long documents
- **Web Interface**: User-friendly interface for searching and uploading documents

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up OpenAI Account](#set-up-openai-account)
  - [Set Up Pinecone Account](#set-up-pinecone-account)
  - [Environment Configuration](#environment-configuration)
  - [Install Dependencies](#install-dependencies)
- [Running the Application](#running-the-application)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Usage](#usage)
  - [Searching Documents](#searching-documents)
  - [Uploading Documents](#uploading-documents)
- [Project Structure](#project-structure)
- [Additional Information](#additional-information)

## Prerequisites

- Python 3.8+ (with pip)
- Node.js 14+ (with npm)
- Git

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/lennox55555/mary-ferrell-foundation-rag.git
cd mary-ferrell-foundation-rag
```

### Set Up OpenAI Account

1. Visit [OpenAI Platform](https://platform.openai.com/) and sign up for an account if you don't have one.
2. Create an API key:
   - Navigate to the [API Keys](https://platform.openai.com/account/api-keys) section in your account
   - Click "Create new secret key"
   - Save this key securely as you'll need it for the `.env` file

### Set Up Pinecone Account

1. Visit [Pinecone](https://www.pinecone.io/) and sign up for an account.
2. Create a new project:
   - From the dashboard, click "Create Project"
   - Name your project (e.g., "mary-ferrell-foundation")
   - Select the "Starter" plan (or higher if needed)

3. Create a new index:
   - In your project, click "Create Index"
   - Name your index "mff" (or choose another name)
   - Set dimensions to 1536 (for OpenAI embeddings)
   - Select "cosine" as the metric
   - Choose the closest region to your location

4. Get your API key:
   - Go to "API Keys" in the left sidebar
   - Copy your API key
   - Note your index name and Pinecone service host URL (looks like "mff-xxxxx.svc.xxx.pinecone.io")

### Environment Configuration

Create a `.env` file in the root directory with the following content:

```
# OpenAI credentials
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone credentials
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=mff
PINECONE_NAMESPACE=documents

# Pinecone host (update this with your actual host from Pinecone dashboard)
PINECONE_HOST=https://mff-xxxxx.svc.xxx.pinecone.io

# RAG settings
SIMILARITY_THRESHOLD=0.6
```

Replace the placeholder values with your actual API keys and Pinecone host.

### Install Dependencies

#### Backend

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for PDF processing)
# On macOS:
brew install tesseract

# On Ubuntu/Debian:
# sudo apt-get install tesseract-ocr

# On Windows:
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
```

#### Frontend

```bash
# Navigate to the frontend directory
cd front-end

# Install Node.js dependencies
npm install

# Return to root directory
cd ..
```

## Running the Application

### Backend

```bash
# From the root directory, with virtual environment activated
python api.py
```

The backend API will be available at http://localhost:3001

### Frontend

```bash
# In a new terminal, navigate to the frontend directory
cd front-end

# Start the React development server
npm start
```

The frontend will be available at http://localhost:3000

## Usage

### Searching Documents

1. Open the application in your web browser
2. Use the "Search" tab (default)
3. Enter your query in the search box
4. Optionally enable/disable citations and similarity metrics
5. Click "Search" to retrieve relevant documents

### Uploading Documents

1. Click on the "Upload" tab
2. Select a file to upload (PDF, CSV, or JSON)
3. Configure processing options if needed:
   - Chunk Size: Controls the size of text chunks (default: 1000 tokens)
   - Chunk Overlap: Controls the overlap between chunks (default: 100 tokens)
   - Clear Index: Check this to replace all existing documents
4. Click "Upload File" to process and index the documents

#### File Format Requirements

- **PDF Files**: Will be processed using OCR if needed
- **CSV Files**: Must include "title" and "contents" (or "text") columns
- **JSON Files**: Must include "title" and "text" fields (or similar variants)

## Project Structure

The application has been consolidated into several key components:

- `api.py`: FastAPI web server that handles queries and file uploads
- `document_processor.py`: Handles reading, extracting, and chunking of documents
- `embedding_service.py`: Creates embeddings and manages the vector database
- `rag_pipeline.py`: Core RAG functionality for retrieving documents
- `front-end/`: React frontend application

## Additional Information

### Generating Document Summaries

You can generate summaries for documents in your database using the `generate_summaries.py` script. This is useful for creating concise overviews of lengthy documents for quicker review.

```bash
# From the root directory, with virtual environment activated
python generate_summaries.py --namespace documents --output summaries.json
```

Options:
- `--namespace`: The Pinecone namespace to use (default: "documents")
- `--output`: Output file path for the summaries (default: "summaries.json")
- `--batch-size`: Number of documents to process in each batch (default: 10)
- `--model`: OpenAI model to use for summarization (default: "gpt-3.5-turbo")

Note: You will need to create an OpenAI developer account as mentioned in the [Set Up OpenAI Account](#set-up-openai-account) section above. The script uses the OpenAI API to generate summaries, which will consume API credits based on the number and length of documents processed.

### Development Mode

By default, the backend runs with hot-reloading enabled, which is convenient for development but should be disabled in production.

### Working with Large Documents

When uploading large PDFs or documents with many pages, processing may take some time. The application will show a loading indicator during processing.

### Logs

The application generates log files in the `out/` directory:
- `api.log`: API server logs
- `main.log`: Command-line interface logs
- `rag_pipeline.log`: RAG functionality logs
- `embedding_service.log`: Embedding and vector database logs
- `document_processor.log`: Document processing logs

Each log file is automatically rotated when it reaches 10MB, with up to 5 backup files kept.

### Troubleshooting

- If you encounter issues with PDF processing, ensure Tesseract OCR is correctly installed
- If embeddings fail, check your OpenAI API key and ensure it has sufficient credits
- For Pinecone connection issues, verify your API key and host URL in the .env file
- Check the log files in the `out/` directory for detailed error information