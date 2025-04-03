# main_processor.py
import os
import argparse
import json
import csv
import requests
from typing import List, Dict, Any
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import openai
import traceback

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

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")
openai.api_key = openai_api_key

# Pinecone API settings
pinecone_api_key = os.getenv("PINECONE_API_KEY", "pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "documents")
pinecone_host = "https://mff-foxube2.svc.aped-4627-b74a.pinecone.io"  # Your actual Pinecone host

class PineconeRESTClient:
    """Simple Pinecone client using REST API instead of Python client"""
    
    def __init__(self, api_key, host, index_name):
        self.api_key = api_key
        self.host = host  # Direct host URL
        self.index_name = index_name
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def describe_index_stats(self):
        """Get index stats"""
        url = f"{self.host}/describe_index_stats"
        response = requests.post(url, headers=self.headers, json={})
        response.raise_for_status()
        return response.json()
    
    def delete_all(self, namespace):
        """Delete all vectors in namespace"""
        url = f"{self.host}/vectors/delete"
        payload = {
            "deleteAll": True,
            "namespace": namespace
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def upsert(self, vectors, namespace):
        """Upsert vectors to index"""
        url = f"{self.host}/vectors/upsert"
        
        # Format vectors for API
        formatted_vectors = []
        for vector_tuple in vectors:
            vector_id, embedding, metadata = vector_tuple
            formatted_vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        payload = {
            "vectors": formatted_vectors,
            "namespace": namespace
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

# Initialize Pinecone client with direct host URL
try:
    pinecone_client = PineconeRESTClient(
        api_key=pinecone_api_key,
        host=pinecone_host,
        index_name=pinecone_index_name
    )
    
    # Test connection by getting index stats
    stats = pinecone_client.describe_index_stats()
    logger.info(f"Connected to Pinecone index. Stats: {stats}")
    
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    logger.error(traceback.format_exc())
    raise

class TextChunker:
    def __init__(self, max_tokens=7000, overlap=50):
        """Initialize text chunker with token limits."""
        self.max_tokens = max_tokens
        self.overlap = overlap
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count using a simple word count approximation."""
        return len(text.split())
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks based on token limits."""
        words = text.split()
        token_count = len(words)
        
        # If text is too large, warn and truncate
        max_allowed_tokens = 4000  # Reduced from any higher value
        if token_count > max_allowed_tokens * 2:
            logger.warning(f"Text is very large: {token_count} tokens. Truncating to {max_allowed_tokens * 2} tokens")
            words = words[:max_allowed_tokens * 2]
            token_count = len(words)
        
        # If text is short enough, return as a single chunk
        if token_count <= self.max_tokens:
            return [{
                "text": " ".join(words),
                "token_count": token_count
            }]
        
        chunks = []
        words_per_chunk = self.max_tokens - self.overlap
        
        for i in range(0, token_count, words_per_chunk):
            chunk_words = words[i:i + self.max_tokens]
            chunk_text = " ".join(chunk_words)
            chunk_token_count = len(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "token_count": chunk_token_count
            })
        
        return chunks

def clear_pinecone_index():
    """Clear all vectors in the Pinecone namespace."""
    try:
        # Check if the namespace has vectors
        stats = pinecone_client.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Delete all vectors in the namespace
        if pinecone_namespace in stats.get("namespaces", {}):
            pinecone_client.delete_all(namespace=pinecone_namespace)
            logger.info(f"Cleared Pinecone index '{pinecone_index_name}' namespace '{pinecone_namespace}'")
        else:
            logger.info(f"Namespace '{pinecone_namespace}' doesn't exist or is empty")
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def embed_text(text: str, retries: int = 3, delay: int = 5) -> List[float]:
    """Generate embedding for the text using OpenAI with extremely aggressive truncation."""
    # Force truncate to maximum 250 words before even trying
    words = text.split()
    if len(words) > 250:
        logger.info(f"Pre-emptively truncating text from {len(words)} words to 250 words")
        text = " ".join(words[:250])
    
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            if "maximum context length" in str(e) and len(text.split()) > 100:
                # If still too long, truncate drastically and retry immediately
                words = text.split()
                text = " ".join(words[:100])
                logger.warning(f"Text still too long for embedding API, truncated to 100 words")
                continue
            elif attempt < retries - 1:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to embed text after {retries} attempts: {str(e)}")
                raise

def process_json_file(file_path: str, chunker: TextChunker, title_field="title", content_field="text", 
                    key_field="key", source_field="source") -> List[Dict[str, Any]]:
    """Process a JSON file and return its chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle single document or list of documents
        if isinstance(data, dict):
            data = [data]
            
        total_docs = len(data)
        logger.info(f"Processing {file_path}: {total_docs} total documents")
        
        # Display data structure for the first entry to help debugging
        if isinstance(data, list) and len(data) > 0:
            logger.info(f"Available fields in first document: {list(data[0].keys())}")
        
        document_chunks = []
        skipped_count = 0
        
        for doc_idx, entry in enumerate(data):
            # Extract fields with fallbacks
            title = entry.get(title_field, "Unknown Title")
            content = entry.get(content_field, "")
            key = entry.get(key_field, f"doc_{doc_idx}")
            source = entry.get(source_field, file_path)
            
            # Skip empty documents
            if not content:
                skipped_count += 1
                continue
                
            # Create full text with title
            full_text = f"{title}: {content}"
            
            # Chunk the text
            chunked = chunker.chunk_text(full_text)
            
            # Add metadata to each chunk
            for chunk in chunked:
                chunk["metadata"] = {
                    "doc_key": key,
                    "doc_title": title,
                    "source": source,
                    "file_path": file_path
                }
                document_chunks.append(chunk)
        
        if document_chunks:
            logger.info(f"SUCCESS: Processed {file_path} - Created {len(document_chunks)} chunks from {total_docs - skipped_count} documents (Skipped {skipped_count} empty documents)")
        else:
            logger.warning(f"FAILURE: No valid documents found in {file_path} - All {total_docs} documents were skipped")
            
        return document_chunks
    except Exception as e:
        logger.error(f"ERROR: Failed to process {file_path}: {str(e)}")
        return []

def process_csv_file(file_path: str, chunker: TextChunker, title_field="title", content_field="contents", 
                    key_field="key", source_field="link") -> List[Dict[str, Any]]:
    """Process a CSV file and return its chunks."""
    logger.info(f"Processing CSV file: {file_path}")
    
    try:
        document_chunks = []
        total_docs = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            rows = list(csv_reader)
            
        total_docs = len(rows)
        logger.info(f"Total rows in CSV {file_path}: {total_docs}")
        
        for doc_idx, row in enumerate(rows):
            # Extract fields with fallbacks
            title = row.get(title_field, "Unknown Title")
            content = row.get(content_field, "")
            key = row.get(key_field, f"csv_doc_{doc_idx}")
            source = row.get(source_field, file_path)
            
            # Skip empty documents
            if not content:
                logger.warning(f"Skipping empty document: {key} in {file_path}")
                continue
                
            # Create full text with title
            full_text = f"{title}: {content}"
            
            # Chunk the text
            chunked = chunker.chunk_text(full_text)
            
            # Add metadata to each chunk
            for chunk in chunked:
                chunk["metadata"] = {
                    "doc_key": key,
                    "doc_title": title,
                    "source": source,
                    "file_path": file_path
                }
                document_chunks.append(chunk)
                
            logger.info(f"Document {doc_idx+1}/{total_docs} (Key: {key}) in {file_path}: Generated {len(chunked)} chunks")
            
        return document_chunks
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def process_crypto_json_file(file_path: str, chunker: TextChunker) -> List[Dict[str, Any]]:
    """Process a cryptpseudo JSON file with special handling."""
    logger.info(f"Processing crypto JSON file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_docs = len(data)
        logger.info(f"Total documents in {file_path}: {total_docs}")
        
        document_chunks = []
        
        for doc_idx, entry in enumerate(data):
            # Extract fields with proper handling for crypto format
            title = entry.get("title", "Unknown Title")
            content = entry.get("contents", "")
            key = entry.get("key", f"crypto_doc_{doc_idx}")
            source = entry.get("link", file_path)
            
            # Skip empty documents
            if not content:
                logger.warning(f"Skipping empty document: {key} in {file_path}")
                continue
                
            # Create full text with title
            full_text = f"{title}: {content}"
            
            # Chunk the text
            chunked = chunker.chunk_text(full_text)
            
            # Add metadata to each chunk
            for chunk in chunked:
                chunk["metadata"] = {
                    "doc_key": key,
                    "doc_title": title,
                    "source": source,
                    "file_path": file_path,
                    "doc_type": entry.get("type", "unknown")
                }
                document_chunks.append(chunk)
                
            logger.info(f"Document {doc_idx+1}/{total_docs} (Key: {key}) in {file_path}: Generated {len(chunked)} chunks")
            
        return document_chunks
    except Exception as e:
        logger.error(f"Error processing crypto JSON file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def upload_chunks_to_pinecone(chunks: List[Dict[str, Any]], batch_size: int = 5):
    """Upload chunks to Pinecone in batches with extreme truncation."""
    if not chunks:
        logger.warning("No chunks to upload")
        return
        
    logger.info(f"Uploading {len(chunks)} chunks to Pinecone")
    
    total_uploaded = 0
    total_failed = 0
    
    # Process in even smaller batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:min(i + batch_size, len(chunks))]
        vectors_to_upload = []
        
        # Prepare batch
        for chunk_idx, chunk in enumerate(batch):
            try:
                # Always truncate text to maximum 250 words
                words = chunk["text"].split()
                truncated_text = " ".join(words[:min(len(words), 250)])
                
                # Get embedding with truncated text
                embedding = embed_text(truncated_text)
                
                # Create a unique vector ID
                key = chunk["metadata"].get("doc_key", f"chunk_{i+chunk_idx}")
                doc_title = chunk["metadata"].get("doc_title", "Unknown")
                vector_id = f"{key}_{i+chunk_idx}_{int(time.time())}"
                
                # Prepare metadata (limit size to prevent Pinecone errors)
                metadata = {
                    "doc_key": key[:100] if key else f"chunk_{i+chunk_idx}",
                    "doc_title": doc_title[:100] if doc_title else "Unknown",
                    "source": chunk["metadata"].get("source", "")[:500],
                    # Store a shorter preview in metadata
                    "text_preview": truncated_text[:500],
                    "original_length": len(chunk["text"])
                }
                
                # Add to batch
                vectors_to_upload.append((vector_id, embedding, metadata))
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+chunk_idx} for embedding: {str(e)}")
                total_failed += 1
                continue
        
        # Upload batch
        if vectors_to_upload:
            try:
                pinecone_client.upsert(vectors=vectors_to_upload, namespace=pinecone_namespace)
                total_uploaded += len(vectors_to_upload)
                logger.info(f"Uploaded batch of {len(vectors_to_upload)} vectors (Total: {total_uploaded}/{len(chunks)})")
            except Exception as e:
                logger.error(f"Error uploading batch to Pinecone: {str(e)}")
                logger.error(traceback.format_exc())
                total_failed += len(vectors_to_upload)
        
        # Add delay to prevent rate limiting
        time.sleep(1)
    
    logger.info(f"Upload complete: {total_uploaded} successful, {total_failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Process documents and save embeddings to Pinecone")
    parser.add_argument("--save_embeddings", action="store_true", help="Save embeddings to Pinecone")
    parser.add_argument("--clear_pinecone", action="store_true", help="Clear Pinecone index before uploading")
    args = parser.parse_args()

    # Clear Pinecone index if requested
    if args.save_embeddings and args.clear_pinecone:
        clear_pinecone_index()


    chunker = TextChunker(max_tokens=1000, overlap=50)
    
    # List of files to process
    files_to_process = [
        # Standard JSON files
        {
            "path": "data/EssaySampleText.json", 
            "type": "json", 
            "processor": process_json_file,
            "params": {
                "title_field": "title", 
                "content_field": "text", 
                "key_field": "key", 
                "source_field": "source"
            }
        },
        # PDF extracted JSON
        {
            "path": "data/pdf_extracted.json", 
            "type": "json", 
            "processor": process_json_file,
            "params": {
                "title_field": "title", 
                "content_field": "text", 
                "key_field": "key", 
                "source_field": "source"
            }
        },
        # Crypto JSON file
        {
            "path": "data/cryptpseudo-mff-duke.json", 
            "type": "crypto_json", 
            "processor": process_crypto_json_file,
            "params": {}
        },
        # CSV file
        {
            "path": "data/dpwdb-mff-duke.csv", 
            "type": "csv", 
            "processor": process_csv_file,
            "params": {
                "title_field": "title", 
                "content_field": "contents", 
                "key_field": "key", 
                "source_field": "link"
            }
        },
    ]

    # Process each file
    processed_files = 0
    failed_files = 0
    all_chunks = []
    
    for file_info in files_to_process:
        file_path = file_info["path"]
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"SKIPPED: File not found - {file_path}")
            failed_files += 1
            continue
            
        # Process file
        processor = file_info["processor"]
        params = file_info["params"]
        
        file_chunks = processor(file_path, chunker, **params)
        
        if file_chunks:
            all_chunks.extend(file_chunks)
            processed_files += 1
        else:
            failed_files += 1
    
    # Upload to Pinecone if requested
    if args.save_embeddings and all_chunks:
        logger.info(f"Uploading {len(all_chunks)} chunks to Pinecone...")
        upload_chunks_to_pinecone(all_chunks)
        logger.info(f"Upload complete!")
    
    # Print final summary
    logger.info(f"== PROCESSING SUMMARY ==")
    logger.info(f"Total files processed: {processed_files}/{len(files_to_process)}")
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    logger.info(f"Failed/skipped files: {failed_files}")
    
    if not all_chunks:
        logger.warning("WARNING: No chunks were generated from any files!")

if __name__ == "__main__":
    main()