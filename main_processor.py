# main_processor.py
import os
import argparse
import json
from typing import List, Dict, Any
import logging
import time
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from text_chunker import TextChunker
from pathlib import Path

# Configure logging to file for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

pinecone_api_key = os.getenv("PINECONE_API_KEY", "pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "documents")

try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    raise

def clear_pinecone_index():
    """Clear all vectors in the Pinecone namespace."""
    try:
        stats = index.describe_index_stats()
        namespace_exists = pinecone_namespace in stats["namespaces"]
        if namespace_exists and stats["namespaces"][pinecone_namespace].get("vector_count", 0) > 0:
            index.delete(delete_all=True, namespace=pinecone_namespace)
            logger.info(f"Cleared Pinecone index '{pinecone_index_name}' namespace '{pinecone_namespace}'")
        else:
            logger.info(f"Namespace '{pinecone_namespace}' is empty or doesn’t exist")
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {str(e)}")
        raise

def embed_text(text: str, retries: int = 3, delay: int = 5) -> List[float]:
    """Generate embedding for the text using OpenAI with retry logic."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to embed text after {retries} attempts: {str(e)}")
                raise

def process_and_upload(chunks: List[Dict[str, Any]], file_path: str):
    """Process and upload chunks to Pinecone incrementally."""
    total_chunks = 0
    skipped_chunks = 0
    vectors_to_upload = []

    batch_size = 50  # Smaller batch size to reduce memory usage

    for doc_idx, doc_chunks in enumerate(chunks):
        for chunk_idx, chunk in enumerate(doc_chunks):
            chunk_text = chunk["text"]
            token_count = chunk["token_count"]

            if token_count > 8192:  # OpenAI embedding limit
                logger.warning(f"Skipping chunk {chunk_idx} for doc {doc_idx} in {file_path}: {token_count} tokens exceeds 8192")
                skipped_chunks += 1
                continue

            try:
                embedding = embed_text(chunk_text)
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk_idx} for doc {doc_idx} in {file_path}: {str(e)}")
                skipped_chunks += 1
                continue

            key = chunk["metadata"]["doc_key"]
            vector_id = f"{key}-{doc_idx}-{chunk_idx}"
            metadata = {
                "doc_index": doc_idx,
                "chunk_id": chunk_idx,
                "doc_key": key,
                "doc_title": chunk["metadata"]["doc_title"],
                "source": chunk["metadata"]["source"],
                "text": chunk_text,
                "token_count": token_count
            }
            vectors_to_upload.append((vector_id, embedding, metadata))
            total_chunks += 1

            if len(vectors_to_upload) >= batch_size:
                try:
                    index.upsert(vectors=vectors_to_upload, namespace=pinecone_namespace)
                    logger.info(f"Uploaded batch of {len(vectors_to_upload)} vectors from {file_path}")
                    vectors_to_upload = []
                    time.sleep(1)  # Throttle to avoid overwhelming Pinecone
                except Exception as e:
                    logger.error(f"Error uploading batch from {file_path}: {str(e)}")
                    raise

    if vectors_to_upload:
        try:
            index.upsert(vectors=vectors_to_upload, namespace=pinecone_namespace)
            logger.info(f"Uploaded final batch of {len(vectors_to_upload)} vectors from {file_path}")
        except Exception as e:
            logger.error(f"Error uploading final batch from {file_path}: {str(e)}")
            raise

    logger.info(f"Processed {file_path}: {total_chunks} chunks embedded, {skipped_chunks} chunks skipped")

def process_file(file_info: Dict[str, str], chunker: TextChunker) -> List[List[Dict[str, Any]]]:
    """Process a single file and return its chunks."""
    file_path = file_info["path"]
    file_type = file_info["type"]
    title_field = file_info["title_field"]
    content_field = file_info["content_field"]
    key_field = file_info["key_field"]
    source_field = file_info["source_field"]

    logger.info(f"Processing file: {file_path} (Type: {file_type})")

    if file_type == "json":
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):  # Single document (PDF JSON)
                data = [data]
            total_docs = len(data)
            logger.info(f"Total documents in {file_path}: {total_docs}")
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {str(e)}")
            return []

        chunks = []
        for doc_idx, entry in enumerate(data):
            title = entry.get(title_field, "")
            content = entry.get(content_field, "")
            key = entry.get(key_field, f"doc_{doc_idx}")
            source = entry.get(source_field, "")
            text = f"{title}: {content}"

            chunked = chunker.chunk_text(text)
            for chunk in chunked:
                chunk["metadata"] = {
                    "doc_key": key,
                    "doc_title": title,
                    "source": source
                }
            chunks.append(chunked)
            logger.info(f"Document {doc_idx+1}/{total_docs} (Key: {key}) in {file_path}: Generated {len(chunked)} chunks")

        return chunks
    return []

def main():
    parser = argparse.ArgumentParser(description="Process documents and save embeddings to Pinecone")
    parser.add_argument("--save_embeddings", action="store_true", help="Save embeddings to Pinecone")
    parser.add_argument("--clear_pinecone", action="store_true", help="Clear Pinecone index before uploading")
    args = parser.parse_args()

    # Define files and directories to process
    files_to_process = [
        {"path": "data/EssaySampleText.json", "type": "json", "title_field": "title", "content_field": "text", "key_field": "key", "source_field": "source"}
    ]
    pdf_extracted_dir = Path("data/pdf_extracted")
    
    if pdf_extracted_dir.exists() and pdf_extracted_dir.is_dir():
        pdf_json_files = list(pdf_extracted_dir.glob("*.json"))
        for json_file in pdf_json_files:
            files_to_process.append({
                "path": str(json_file),
                "type": "json",
                "title_field": "title",
                "content_field": "text",
                "key_field": "key",
                "source_field": "source"
            })
    else:
        logger.warning("Directory 'data/pdf_extracted' not found. Only processing EssaySampleText.json")

    total_docs_processed = 0

    # Clear Pinecone index if requested
    if args.save_embeddings and args.clear_pinecone:
        clear_pinecone_index()

    # Initialize chunker
    chunker = TextChunker(max_tokens=7000, overlap=50)

    # Process files one-by-one
    for file_info in files_to_process:
        file_chunks = process_file(file_info, chunker)
        if not file_chunks:
            continue
        total_docs_processed += len(file_chunks)

        if args.save_embeddings:
            process_and_upload(file_chunks, file_info["path"])
            time.sleep(2)  # Throttle between files to reduce system load

    logger.info(f"Completed processing {total_docs_processed} documents across all files")

if __name__ == "__main__":
    main()