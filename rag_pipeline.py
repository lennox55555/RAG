import os
from dotenv import load_dotenv  # Add this import
import openai
from typing import Dict, List, Any
import logging

# Load environment variables from .env file
load_dotenv()  # Add this line

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

# Initialize Pinecone client using the new Pinecone class
try:
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if the index exists; if not, create it.
    if pinecone_index_name not in pc.list_indexes().names():
        logger.info(f"Index '{pinecone_index_name}' not found. Creating new index...")
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,  # Dimension for OpenAI embeddings (text-embedding-ada-002)
            metric='cosine',
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-west-2")
            )
        )
    index = pc.Index(pinecone_index_name)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    raise

class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline without a similarity threshold."""
        logger.info("RAGPipeline initialized without similarity threshold filtering.")
        self.index = index
        self.namespace = pinecone_namespace

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for the query using OpenAI."""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=query
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def retrieve_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Retrieve all relevant documents from Pinecone without filtering by similarity threshold."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            # Return all matches without filtering.
            return results["matches"]
        except Exception as e:
            logger.error(f"Error retrieving documents from Pinecone: {str(e)}")
            return []

    def generate_response(self, query: str, documents: List[str]) -> str:
        """Generate a response using OpenAI based on retrieved documents."""
        try:
            # Truncate each document to avoid context overflow (e.g., 1000 characters per doc)
            truncated_docs = [doc[:1000] for doc in documents]
            prompt = f"Query: {query}\nDocuments: {' '.join(truncated_docs)}\nProvide a concise answer:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"].strip()
        except openai.error.InvalidRequestError as e:
            logger.error(f"OpenAI context error: {str(e)}")
            return "I encountered an error processing the documents. Please try a simpler query."
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def process_query(self, query: str, with_citations: bool = False) -> Dict[str, Any]:
        """Process the query and return a response with optional citations."""
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embed_query(query)

            # Step 2: Retrieve all documents (no threshold filtering)
            matches = self.retrieve_documents(query_embedding)
            num_docs = len(matches)

            # Step 3: Prepare citations and document texts
            citations = {}
            documents_text = []
            if matches:
                for i, match in enumerate(matches):
                    doc_text = match["metadata"]["text"]
                    documents_text.append(doc_text)
                    if with_citations:
                        citations[str(i)] = {
                            "title": match["metadata"].get("doc_title", "Unknown"),
                            "source": match["metadata"].get("source", ""),
                            "similarity": match.get("score")
                        }

            # Step 4: Generate a response using the retrieved documents
            response = self.generate_response(query, documents_text) if documents_text else "I don't have enough information to answer."

            # Step 5: Calculate confidence using the top match score (if any)
            confidence_score = matches[0]["score"] if matches else 0.0
            confidence = {
                "score": confidence_score,
                "is_relevant": num_docs > 0
            }

            return {
                "query": query,
                "response": response,
                "citations": citations if with_citations else None,
                "num_docs_retrieved": num_docs,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "query": query,
                "response": "I encountered an error while generating a response. Please try again.",
                "citations": None,
                "num_docs_retrieved": 0,
                "confidence": {"score": 0.0, "is_relevant": False}
            }

def create_rag_pipeline_from_env() -> RAGPipeline:
    """Create and return a RAGPipeline instance without using similarity threshold filtering."""
    return RAGPipeline()

if __name__ == "__main__":
    # For testing purposes
    pipeline = create_rag_pipeline_from_env()
    result = pipeline.process_query("Who is JFK and how did he die?")
    print(result)
