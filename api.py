# import fastapi components
from fastapi import FastAPI, HTTPException, Depends, Query
# import cors middleware
from fastapi.middleware.cors import CORSMiddleware
# import base model and field
from pydantic import BaseModel, Field
# import typing modules
from typing import Dict, Any, List, Optional, Union
# import os module
import os
# import uvicorn server
import uvicorn
# import traceback for error logging
import traceback
# import logging module
import logging

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import rag pipeline
try:
    from rag_pipeline import create_rag_pipeline_from_env, RAGPipeline
except Exception as e:
    logger.error(f"Error importing RAG pipeline: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# define request model
class QueryRequest(BaseModel):
    query: str
    with_citations: bool = False
    similarity_threshold: Optional[float] = Field(None, description="Override default similarity threshold (0.0-1.0)")

# define confidence info model
class ConfidenceInfo(BaseModel):
    score: float
    is_relevant: bool

# define citation model
class Citation(BaseModel):
    title: str
    source: str
    similarity: Optional[float] = None

# define query response model
class QueryResponse(BaseModel):
    query: str
    response: str
    citations: Optional[Dict[str, Citation]] = None
    num_docs_retrieved: int = 0
    confidence: ConfidenceInfo

# define health response model
class HealthResponse(BaseModel):
    status: str
    version: str = "1.1.0"

# initialize fastapi app
app = FastAPI(
    title="RAG System API",
    description="API for Retrieval-Augmented Generation system",
    version="1.1.0"
)

# add cors support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# dependency to get rag pipeline
def get_rag_pipeline(similarity_threshold: Optional[float] = None):
    try:
        logger.info("Creating RAG pipeline")
        
        # check for custom threshold
        if similarity_threshold is not None:
            # validate threshold value
            if not 0 <= similarity_threshold <= 1:
                raise HTTPException(status_code=400, detail="Similarity threshold must be between 0 and 1")
            
            os.environ["SIMILARITY_THRESHOLD"] = str(similarity_threshold)
        
        return create_rag_pipeline_from_env()
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG pipeline: {str(e)}")

# health check route
@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "version": "1.1.0"}

# query route
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, 
    pipeline: RAGPipeline = Depends(lambda: get_rag_pipeline(None))
):
    logger.info(f"Processing query: {request.query}")
    
    # override pipeline if threshold is provided
    if request.similarity_threshold is not None:
        pipeline = get_rag_pipeline(request.similarity_threshold)
    
    try:
        result = pipeline.process_query(request.query, request.with_citations)
        
        # parse citations
        citations_model = None
        if result.get("citations"):
            citations_model = {}
            for k, v in result["citations"].items():
                citations_model[str(k)] = Citation(
                    title=v.get("title", "Unknown"),
                    source=v.get("source", ""),
                    similarity=v.get("similarity", None)
                )
        
        # prepare confidence info
        confidence = ConfidenceInfo(
            score=result["confidence"]["score"],
            is_relevant=result["confidence"]["is_relevant"]
        )
        
        logger.info(f"Query processed with confidence {confidence.score:.2f}, retrieved {result['num_docs_retrieved']} documents")
        
        return {
            "query": result["query"],
            "response": result["response"],
            "citations": citations_model,
            "num_docs_retrieved": result["num_docs_retrieved"],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# debug route
@app.get("/debug")
async def debug():
    try:
        # create pipeline
        pipeline = get_rag_pipeline()
        
        # test queries
        test_queries = [
            "What happened to JFK?",
            "Who was Lee Harvey Oswald?",
            "completely unrelated query about bananas",
            "poop"
        ]
        
        results = {}
        for query in test_queries:
            query_result = pipeline.process_query(query, with_citations=False)
            results[query] = {
                "num_docs": query_result["num_docs_retrieved"],
                "confidence": query_result["confidence"],
                "response_snippet": query_result["response"][:100] + "..." if query_result["response"] else ""
            }
        
        # return debug results
        return {
            "similarity_threshold": pipeline.similarity_threshold,
            "test_results": results,
            "environment": {
                "SIMILARITY_THRESHOLD": os.getenv("SIMILARITY_THRESHOLD", "Not set"),
                "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
                "has_pinecone_key": bool(os.getenv("PINECONE_API_KEY")),
                "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "Not set")
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# get current similarity threshold
@app.get("/settings/similarity-threshold")
async def get_similarity_threshold():
    try:
        # read threshold from env
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
        return {"similarity_threshold": threshold}
    except Exception as e:
        logger.error(f"Error retrieving similarity threshold: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving similarity threshold: {str(e)}")

# update similarity threshold
@app.put("/settings/similarity-threshold")
async def update_similarity_threshold(threshold: float = Query(..., ge=0.0, le=1.0)):
    try:
        # set threshold in env
        os.environ["SIMILARITY_THRESHOLD"] = str(threshold)
        logger.info(f"Updated similarity threshold to {threshold}")
        return {"similarity_threshold": threshold, "status": "updated"}
    except Exception as e:
        logger.error(f"Error updating similarity threshold: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating similarity threshold: {str(e)}")

# entrypoint function
def main():
    # get port number
    port = int(os.getenv("PORT", "3001"))
    
    # check openai key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
    
    # check pinecone key
    if not os.getenv("PINECONE_API_KEY"):
        logger.warning("PINECONE_API_KEY environment variable not set, using default value")
    
    logger.info(f"Starting server on port {port}")
    
    # run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Set to False in production
    )

# run main function
if __name__ == "__main__":
    main()
