# import required modules
import os
import time
from typing import List, Dict, Any, Optional
import uuid
import json
import numpy as np
from tqdm import tqdm
import openai
from pinecone import Pinecone

# define embedding creator class
class EmbeddingCreator:
    def __init__(self, 
                 pinecone_api_key: str,
                 pinecone_index_name: str,
                 pinecone_namespace: str = "documents",
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-ada-002",
                 batch_size: int = 32,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        # initialize pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.namespace = pinecone_namespace
        
        # set openai key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # set config parameters
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    # get embeddings for a batch
    def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        for attempt in range(self.max_retries):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=texts
                )
                return [data["embedding"] for data in response["data"]]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error getting batch embeddings, retrying in {self.retry_delay} seconds: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to get batch embeddings after {self.max_retries} attempts: {str(e)}")
                    return [None for _ in range(len(texts))]
    
    # create embeddings from chunks
    def create_embeddings(self, chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> List[Dict[str, Any]]:
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
                    print(f"Skipped chunk {batch[j]['doc_title']} (chunk_id: {batch[j]['chunk_id']}) due to embedding failure")
            
            time.sleep(0.5)
        
        print(f"Created embeddings for {len(embedded_chunks)} chunks")
        
        # optionally save embeddings
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
    
    # upload embeddings to pinecone
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
    
    # full process and upload pipeline
    def process_and_upload(self, chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        embedded_chunks = self.create_embeddings(chunks, save_path)
        self.upload_to_pinecone(embedded_chunks)

# main execution block
if __name__ == "__main__":
    # import helper modules
    from data_reader import DataReader
    from text_extractor import TextExtractor
    from text_chunker import TextChunker
    
    # get openai key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # set pinecone config values
    pinecone_api_key = "pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp"
    pinecone_index_name = "mff"
    
    # read and extract documents
    reader = DataReader("data/EssaySampleText.json")
    extractor = TextExtractor(reader)
    documents = extractor.extract_all_texts()
    
    # chunk text data
    chunker = TextChunker(max_tokens=7500, chunk_overlap=200)
    chunks = chunker.chunk_documents(documents)
    
    # create and run embedding creator
    embedding_creator = EmbeddingCreator(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key,
        batch_size=32
    )
    
    embedding_creator.process_and_upload(chunks, save_path="data/embeddings.json")
