from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
import os
import uvicorn
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from rag_pipeline import create_rag_pipeline_from_env, RAGPipeline
    from text_similarity import TextSimilarityCalculator
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Define API models
class QueryRequest(BaseModel):
    query: str
    with_citations: bool = False
    include_similarity: bool = False  # New parameter

class ConfidenceInfo(BaseModel):
    score: float
    level: str
    explanation: str

class QueryResponse(BaseModel):
    query: str
    response: str
    citations: Optional[Dict[Union[str, int], Any]] = None
    num_docs_retrieved: int = 0
    retrieved_docs: Optional[List[Dict[str, Any]]] = None  # New field for documents with similarity
    confidence: Optional[ConfidenceInfo] = None  # New field for confidence info

class HealthResponse(BaseModel):
    status: str

# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for Retrieval-Augmented Generation system",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for RAG pipeline
def get_rag_pipeline():
    try:
        logger.info("Creating RAG pipeline")
        return create_rag_pipeline_from_env()
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG pipeline: {str(e)}")

# Dependency for similarity calculator
def get_similarity_calculator():
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        return TextSimilarityCalculator(openai_api_key=openai_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize similarity calculator: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to initialize similarity calculator: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, 
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    similarity_calculator: TextSimilarityCalculator = Depends(get_similarity_calculator)
):
    logger.info(f"Processing query: {request.query}")
    try:
        # Get result from pipeline
        result = pipeline.process_query(request.query, request.with_citations)
        
        # Convert any integer citation keys to strings to satisfy Pydantic validation
        if result.get("citations"):
            result["citations"] = {str(k): v for k, v in result["citations"].items()}
        
        # Initialize response data
        response_data = {
            "query": result["query"],
            "response": result["response"],
            "citations": result["citations"],
            "num_docs_retrieved": result["num_docs_retrieved"]
        }
        
        # Add similarity metrics if requested
        if request.include_similarity and hasattr(similarity_calculator, 'process_results_with_similarity'):
            # Get the retrieved docs from the result
            retrieved_docs = result.get("retrieved_docs", [])
            
            if retrieved_docs:
                # Use the similarity calculator to add metrics
                similarity_result = similarity_calculator.process_results_with_similarity(
                    request.query, 
                    retrieved_docs
                )
                
                # Update response with similarity data
                response_data["retrieved_docs"] = similarity_result["results"]
                response_data["confidence"] = similarity_result["confidence"]
        
        logger.info(f"Query processed successfully, retrieved {result['num_docs_retrieved']} documents")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Main function to run the API
def main():
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Check if required API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
    
    if not os.getenv("PINECONE_API_KEY"):
        logger.warning("PINECONE_API_KEY environment variable not set, using default value")
    
    logger.info(f"Starting server on port {port}")
    
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Set to False in production
    )

if __name__ == "__main__":
    main()