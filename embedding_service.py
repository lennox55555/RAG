import os
import time
import uuid
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import openai
import pinecone
import requests
from dotenv import load_dotenv

# setup logging
from logger_config import setup_logger
logger = setup_logger("embedding_service")

# load env vars
load_dotenv()

class EmbeddingService:
    def __init__(self, 
                pinecone_api_key: Optional[str] = None,
                pinecone_index_name: str = "mff",
                pinecone_namespace: str = "documents",
                openai_api_key: Optional[str] = None,
                embedding_model: str = "text-embedding-3-small",
                batch_size: int = 32,
                max_retries: int = 3,
                retry_delay: int = 5):
        # get api keys from env if not provided
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not provided and not found in environment variables")
            
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
            
        # init openai
        openai.api_key = self.openai_api_key
        
        # init pinecone params for direct API calls
        self.index_name = pinecone_index_name
        self.namespace = pinecone_namespace
        self.pinecone_host = "https://mff-foxube2.svc.aped-4627-b74a.pinecone.io"
        self.headers = {
            "Api-Key": self.pinecone_api_key,
            "Content-Type": "application/json"
        }
        
        # set config params
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized EmbeddingService with index: {pinecone_index_name}, namespace: {pinecone_namespace}")
    
    def embed_text(self, text: str) -> List[float]:
        # create embedding for single text
        for attempt in range(self.max_retries):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=text
                )
                return response["data"][0]["embedding"]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Error embedding text, retrying in {self.retry_delay}s: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to embed text after {self.max_retries} attempts: {str(e)}")
                    raise
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        # get embeddings for batch
        for attempt in range(self.max_retries):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=texts
                )
                return [data["embedding"] for data in response["data"]]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Error getting batch embeddings, retrying in {self.retry_delay}s: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to get batch embeddings after {self.max_retries} attempts: {str(e)}")
                    return [None for _ in range(len(texts))]
    
    def create_embeddings(self, chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        # create embeddings for chunks
        embedded_chunks = []
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Creating embeddings"):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk["text"] for chunk in batch]
            
            batch_embeddings = self._get_embeddings_batch(batch_texts)
            
            for j, embedding in enumerate(batch_embeddings):
                if embedding is not None:
                    batch[j]["embedding"] = embedding
                    embedded_chunks.append(batch[j])
                else:
                    logger.warning(f"Skipped chunk {batch[j]['doc_title']} (chunk_id: {batch[j]['chunk_id']}) due to embedding failure")
            
            time.sleep(0.5)  # rate limiting
        
        logger.info(f"Created embeddings for {len(embedded_chunks)} chunks")
        
        # save embeddings if requested
        if save_path:
            serializable_chunks = []
            for chunk in embedded_chunks:
                serializable_chunk = chunk.copy()
                if isinstance(serializable_chunk["embedding"], np.ndarray):
                    serializable_chunk["embedding"] = serializable_chunk["embedding"].tolist()
                serializable_chunks.append(serializable_chunk)
                
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f)
            logger.info(f"Saved embeddings to {save_path}")
        
        return embedded_chunks
    
    def upload_to_pinecone(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        # upload embeddings to pinecone using direct API
        total_uploaded = 0
        
        for i in tqdm(range(0, len(embedded_chunks), self.batch_size), desc="Uploading to Pinecone"):
            batch = embedded_chunks[i:i + self.batch_size]
            
            vectors = []
            for chunk in batch:
                chunk_id = str(uuid.uuid4())
                metadata = {
                    "text": chunk["text"],
                    "doc_title": chunk["doc_title"],
                    "source": chunk["source"],
                    "doc_key": chunk["doc_key"],
                    "page_num": chunk.get("page_num", 0),
                    "total_pages": chunk.get("total_pages", 0)
                }
                
                vectors.append({
                    "id": chunk_id,
                    "values": chunk["embedding"],
                    "metadata": metadata
                })
            
            for attempt in range(self.max_retries):
                try:
                    # Direct API call to Pinecone
                    url = f"{self.pinecone_host}/vectors/upsert"
                    payload = {
                        "vectors": vectors,
                        "namespace": self.namespace
                    }
                    
                    response = requests.post(url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    
                    total_uploaded += len(vectors)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Error uploading to Pinecone, retrying in {self.retry_delay}s: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to upload batch to Pinecone after {self.max_retries} attempts: {str(e)}")
            
            time.sleep(1)  # rate limiting
        
        logger.info(f"Uploaded {total_uploaded} vectors to Pinecone")
    
    def process_and_upload(self, chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        # full embed and upload process
        embedded_chunks = self.create_embeddings(chunks, save_path)
        self.upload_to_pinecone(embedded_chunks)
    
    def clear_pinecone_index(self) -> None:
        # clear all vectors from namespace using direct API
        try:
            # Get stats before deletion
            url = f"{self.pinecone_host}/describe_index_stats"
            response = requests.post(url, headers=self.headers, json={})
            stats = response.json()
            logger.info(f"Current index stats: {stats}")
            
            # Delete all vectors from namespace
            url = f"{self.pinecone_host}/vectors/delete"
            response = requests.post(
                url, 
                headers=self.headers, 
                json={"deleteAll": True, "namespace": self.namespace}
            )
            response.raise_for_status()
            logger.info(f"Cleared all vectors from namespace '{self.namespace}'")
            
            # Verify deletion
            url = f"{self.pinecone_host}/describe_index_stats"
            response = requests.post(url, headers=self.headers, json={})
            stats_after = response.json()
            logger.info(f"Index stats after clearing: {stats_after}")
            
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {str(e)}")
            raise
    
    def retrieve_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        # get similar docs from pinecone using direct API
        try:
            url = f"{self.pinecone_host}/query"
            payload = {
                "namespace": self.namespace,
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            results = response.json()
            
            return results.get("matches", [])
        except Exception as e:
            logger.error(f"Error retrieving similar docs: {str(e)}")
            return []

# test
if __name__ == "__main__":
    # import for testing
    from document_processor import DataReader, TextExtractor, TextChunker
    
    # create embedding service
    embedding_service = EmbeddingService()
    
    # process sample docs
    reader = DataReader("data/EssaySampleText.json")
    extractor = TextExtractor(reader)
    texts = extractor.extract_all_texts()
    
    # create chunks
    chunker = TextChunker(max_tokens=1000, chunk_overlap=100)
    chunks = chunker.chunk_documents(texts)
    
    # process and upload
    embedding_service.process_and_upload(chunks, save_path="data/embeddings.json")
    
    # test query
    test_text = "Who is JFK?"
    query_embedding = embedding_service.embed_text(test_text)
    matches = embedding_service.retrieve_similar(query_embedding, top_k=3)
    
    logger.info(f"Test query: {test_text}")
    logger.info(f"Found {len(matches)} matches")
    for i, match in enumerate(matches):
        logger.info(f"Match {i+1}: Score {match['score']}, Title: {match['metadata']['doc_title']}")
        logger.info(f"Text preview: {match['metadata']['text'][:100]}...")