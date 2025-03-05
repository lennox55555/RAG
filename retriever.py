import os
from typing import List, Dict, Any, Optional, Tuple
import openai
from pinecone import Pinecone

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
                 top_k: int = 5):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.namespace = pinecone_namespace
        
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def _get_query_embedding(self, query: str) -> List[float]:
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting query embedding: {str(e)}")
            return [0.0] * 1536
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self._get_query_embedding(query)
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                namespace=self.namespace,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error querying Pinecone: {str(e)}")
            return []
        
        retrieved_docs = []
        for match in results.matches:
            retrieved_docs.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "doc_title": match.metadata.get("doc_title", ""),
                "source": match.metadata.get("source", ""),
                "chunk_id": match.metadata.get("chunk_id", "")
            })
        
        return retrieved_docs
    
    def retrieve_and_format(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        results = self.retrieve(query)
        
        if not results:
            return [], "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(results):
            context_parts.append(
                f"[Document {i+1}] {doc['text']}\n"
                f"Source: {doc['doc_title']}"
            )
        
        formatted_context = "\n\n".join(context_parts)
        
        return results, formatted_context


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "your-default-pinecone-key")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
    
    retriever = DocumentRetriever(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key
    )
    
    query = "What happened to JFK?"
    results, formatted_context = retriever.retrieve_and_format(query)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} relevant documents")
    print("\nFormatted Context:")
    print(formatted_context[:500] + "...")
