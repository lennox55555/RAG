import os
import time
from typing import List, Dict, Any, Optional
import uuid
import json
import numpy as np
from tqdm import tqdm
import openai
from pinecone import Pinecone

class EmbeddingCreator:
    """
    Class for creating embeddings and uploading to Pinecone.
    """
    def __init__(self, 
                 pinecone_api_key: str,
                 pinecone_index_name: str,
                 pinecone_namespace: str = "documents",
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-ada-002",
                 batch_size: int = 100,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        
        # pine cone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.namespace = pinecone_namespace
        
        # init open ai
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _get_embedding(self, text: str) -> List[float]:
        for attempt in range(self.max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error getting embedding, retrying in {self.retry_delay} seconds: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to get embedding after {self.max_retries} attempts: {str(e)}")
                    return [0.0] * 1536
        
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(self.max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error getting batch embeddings, retrying in {self.retry_delay} seconds: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to get batch embeddings after {self.max_retries} attempts: {str(e)}")
                    return [[0.0] * 1536 for _ in range(len(texts))]
    
    def create_embeddings(self, chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        embedded_chunks = []
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Creating embeddings"):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk["text"] for chunk in batch]
            
            batch_embeddings = self._get_embeddings_batch(batch_texts)
            
            for j, embedding in enumerate(batch_embeddings):
                batch[j]["embedding"] = embedding
                embedded_chunks.append(batch[j])
            
            time.sleep(0.5)
        
        print(f"Created embeddings for {len(embedded_chunks)} chunks")
        
        if save_path:
            serializable_chunks = []
            for chunk in embedded_chunks:
                serializable_chunk = chunk.copy()
                if isinstance(serializable_chunk["embedding"], np.ndarray):
                    serializable_chunk["embedding"] = serializable_chunk["embedding"].tolist()
                serializable_chunks.append(serializable_chunk)
                
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f)
            print(f"Saved embeddings to {save_path}")
        
        return embedded_chunks
    
    def upload_to_pinecone(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        total_uploaded = 0
        
        for i in tqdm(range(0, len(embedded_chunks), self.batch_size), desc="Uploading to Pinecone"):
            batch = embedded_chunks[i:i + self.batch_size]
            
            vectors = []
            for chunk in batch:
                chunk_id = str(uuid.uuid4())
                metadata = {k: v for k, v in chunk.items() if k != "embedding" and not isinstance(v, (list, dict))}
                
                vectors.append({
                    "id": chunk_id,
                    "values": chunk["embedding"],
                    "metadata": metadata
                })
            
            for attempt in range(self.max_retries):
                try:
                    self.index.upsert(
                        vectors=vectors,
                        namespace=self.namespace
                    )
                    total_uploaded += len(vectors)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"Error uploading to Pinecone, retrying in {self.retry_delay} seconds: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        print(f"Failed to upload batch to Pinecone after {self.max_retries} attempts: {str(e)}")
            
            time.sleep(1)
        
        print(f"Successfully uploaded {total_uploaded} vectors to Pinecone")
    
    def process_and_upload(self, chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        embedded_chunks = self.create_embeddings(chunks, save_path)
        self.upload_to_pinecone(embedded_chunks)


if __name__ == "__main__":
    from data_reader import DataReader
    from text_extractor import TextExtractor
    from text_chunker import TextChunker
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    pinecone_api_key = "your-default-pinecone-key"
    pinecone_index_name = "mff"
    
    reader = DataReader("data/EssaySampleText.json")
    extractor = TextExtractor(reader)
    documents = extractor.extract_all_texts()
    
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(documents)
    
    embedding_creator = EmbeddingCreator(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key
    )
    
    embedding_creator.process_and_upload(chunks, save_path="data/embeddings.json")
