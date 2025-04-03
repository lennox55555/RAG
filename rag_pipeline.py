# Import libraries
import os
from dotenv import load_dotenv
import openai
import requests
from typing import Dict, List, Any
import logging
import json
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

# Pinecone configuration variables
pinecone_api_key = os.getenv("PINECONE_API_KEY", "your_actual_api_key")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "documents")
pinecone_host = "https://mff-foxube2.svc.aped-4627-b74a.pinecone.io"  # Use your actual host URL

class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline."""
        logger.info("RAGPipeline initialized.")
        self.host = pinecone_host
        self.api_key = pinecone_api_key
        self.namespace = pinecone_namespace
        self.similarity_threshold = 0.6  # Add a default threshold
        self.headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for the query using OpenAI."""
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",  # Using latest available model
                input=query
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def retrieve_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents from Pinecone using direct API call."""
        try:
            url = f"{self.host}/query"
            payload = {
                "vector": query_embedding,
                "top_k": top_k,
                "namespace": self.namespace,
                "include_metadata": True
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            results = response.json()
            return results.get("matches", [])
        except Exception as e:
            logger.error(f"Error retrieving documents from Pinecone: {str(e)}")
            return []

    def generate_response(self, query: str, documents: List[str]) -> str:
        """Generate a response using OpenAI based on retrieved documents."""
        try:
            # Truncate each document to avoid context overflow (e.g., 1000 characters per doc)
            truncated_docs = [doc[:750] for doc in documents]
            combined_docs = "\n\n---\n\n".join(truncated_docs)
            
            prompt = f"""You are a helpful research assistant for the Mary Ferrell Foundation, which focuses on historical documents related to the JFK assassination, civil rights, and other significant historical events.

Query: {query}

Context from documents:
{combined_docs}

Based only on the provided context, please give a comprehensive and accurate answer to the query. 
If the context doesn't contain relevant information, acknowledge this limitation.
Format your response in a scholarly tone appropriate for historical research. Cite your sources using numbers in square brackets [1] where appropriate."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def process_query(self, query: str, with_citations: bool = False) -> Dict[str, Any]:
        """Process the query and return a response with optional citations."""
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embed_query(query)

            # Step 2: Retrieve documents
            matches = self.retrieve_documents(query_embedding)
            num_docs = len(matches)

            # Step 3: Prepare citations and document texts
            citations = {}
            documents_text = []
            retrieved_docs = []
            
            if matches:
                for i, match in enumerate(matches):
                    similarity_score = match.get("score", 0)
                    
                    # Skip if below threshold (unless we have no results)
                    if similarity_score < self.similarity_threshold and i > 0:
                        continue
                        
                    doc_text = match["metadata"].get("text", "")
                    doc_title = match["metadata"].get("doc_title", "Unknown")
                    doc_source = match["metadata"].get("source", "")
                    
                    documents_text.append(doc_text)
                    
                    # Add to retrieved docs for frontend
                    retrieved_docs.append({
                        "doc_title": doc_title,
                        "source": doc_source,
                        "text": doc_text,
                        "similarity": similarity_score,
                        "similarity_category": self._get_similarity_category(similarity_score)
                    })
                    
                    if with_citations:
                        citations[str(i+1)] = {
                            "title": doc_title,
                            "source": doc_source,
                            "similarity": similarity_score
                        }

            # Step 4: Generate a response using the retrieved documents
            if documents_text:
                response = self.generate_response(query, documents_text)
                # Add confidence level explanation
                confidence_level = self._get_confidence_level(matches[0]["score"] if matches else 0)
            else:
                response = "I don't have specific information about that in my knowledge base. Please ask about a topic covered in the documents I have access to."
                confidence_level = {"score": 0.0, "is_relevant": False, "level": "Very Low", "explanation": "No relevant documents found."}

            # Build the result
            result = {
                "query": query,
                "response": response,
                "retrieved_docs": retrieved_docs if with_citations else None,
                "citations": citations if with_citations else None,
                "num_docs_retrieved": num_docs,
                "confidence": confidence_level
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "query": query,
                "response": "I encountered an error while generating a response. Please try again.",
                "citations": None,
                "num_docs_retrieved": 0,
                "confidence": {"score": 0.0, "is_relevant": False, "level": "Very Low", "explanation": "An error occurred during processing."}
            }
            
    def _get_similarity_category(self, score: float) -> str:
        """Convert similarity score to a category."""
        if score >= 0.9:
            return "Very High"
        elif score >= 0.8:
            return "High"
        elif score >= 0.7:
            return "Moderate"
        elif score >= 0.6:
            return "Low"
        else:
            return "Very Low"
            
    def _get_confidence_level(self, score: float) -> Dict[str, Any]:
        """Convert similarity score to confidence information."""
        category = self._get_similarity_category(score)
        explanations = {
            "Very High": "The response is based on documents with very high relevance to your query.",
            "High": "The response is based on documents with good relevance to your query.",
            "Moderate": "The response is based on documents with moderate relevance to your query.",
            "Low": "The response may be partially relevant but is based on limited matching documents.",
            "Very Low": "The response has low confidence as no strongly relevant documents were found."
        }
        
        return {
            "score": score,
            "is_relevant": score >= self.similarity_threshold,
            "level": category,
            "explanation": explanations[category]
        }

def create_rag_pipeline_from_env() -> RAGPipeline:
    """Create and return a RAGPipeline instance."""
    # Check if similarity threshold is in environment
    threshold = os.getenv("SIMILARITY_THRESHOLD")
    pipeline = RAGPipeline()
    if threshold:
        try:
            pipeline.similarity_threshold = float(threshold)
        except ValueError:
            logger.warning(f"Invalid similarity threshold '{threshold}', using default")
    return pipeline

if __name__ == "__main__":
    # For testing purposes
    pipeline = create_rag_pipeline_from_env()
    result = pipeline.process_query("Who is JFK and how did he die?")
    print(json.dumps(result, indent=2))