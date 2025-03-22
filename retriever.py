import os
from typing import List, Dict, Any, Optional, Tuple
import openai
from pinecone import Pinecone
import numpy as np
from scipy.spatial.distance import cosine

class DocumentRetriever:
    """
    Class for retrieving relevant documents from Pinecone based on user queries.
    """
    def __init__(self, 
                 pinecone_api_key: str,
                 pinecone_index_name: str,
                 pinecone_namespace: str = "documents",
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-ada-002",
                 top_k: int = 5,
                 similarity_threshold: float = 0.6):  # Add similarity threshold
        """
        Initialize the DocumentRetriever.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_index_name: Name of the Pinecone index
            pinecone_namespace: Namespace in the Pinecone index
            openai_api_key: OpenAI API key (if not provided, will use environment variable)
            embedding_model: Name of the embedding model to use
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score for a document to be considered relevant
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.namespace = pinecone_namespace
        
        # Initialize OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold  # Store the threshold
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (1 is most similar, 0 is least similar)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        return 1 - cosine(vec1_np, vec2_np)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            
        Returns:
            List of relevant document dictionaries
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Query Pinecone
            query_results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Process results into standardized format
            results = []
            for match in query_results["matches"]:
                score = match["score"]
                metadata = match["metadata"]
                
                # Only include results above the similarity threshold
                if score >= self.similarity_threshold:
                    result = {
                        "text": metadata.get("text", ""),
                        "doc_title": metadata.get("doc_title", "Unknown"),
                        "source": metadata.get("source", ""),
                        "similarity_score": score
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_and_format(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve and format documents for a query.
        
        Args:
            query: Query text
            
        Returns:
            Tuple of (list of results, formatted context string)
        """
        results = self.retrieve(query)
        
        # Format results as context
        formatted_context = self._format_context(results)
        
        return results, formatted_context
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"Document {i}:\nTitle: {result['doc_title']}\nContent: {result['text']}\n")
        
        return "\n".join(context_parts)