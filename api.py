from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import os
import uuid
import uvicorn
import traceback
import logging
import tempfile
import json
import csv
import shutil
from pathlib import Path
from dotenv import load_dotenv

# setup logging
from logger_config import setup_logger
logger = setup_logger("api")

# load env vars
load_dotenv()

# import rag pipeline
try:
    from rag_pipeline import create_rag_pipeline_from_env, RAGPipeline
    from document_processor import DataReader, TextExtractor, TextChunker, PDFProcessor
    from embedding_service import EmbeddingService
except Exception as e:
    logger.error(f"Error importing RAG pipeline: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# request model
class QueryRequest(BaseModel):
    query: str
    with_citations: bool = False
    similarity_threshold: Optional[float] = Field(None, description="Override default similarity threshold (0.0-1.0)")

# confidence info model
class ConfidenceInfo(BaseModel):
    score: float
    is_relevant: bool
    level: str
    explanation: str

# citation model
class Citation(BaseModel):
    title: str
    source: str
    similarity: Optional[float] = None

# query response model
class QueryResponse(BaseModel):
    query: str
    response: str
    citations: Optional[Dict[str, Citation]] = None
    num_docs_retrieved: int = 0
    confidence: ConfidenceInfo

# health response model
class HealthResponse(BaseModel):
    status: str
    version: str = "1.1.0"

# upload response model
class UploadResponse(BaseModel):
    filename: str
    status: str
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    message: str

# init fastapi app
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

# get rag pipeline
def get_rag_pipeline(similarity_threshold: Optional[float] = None):
    try:
        logger.info("Creating RAG pipeline")
        
        # check for custom threshold
        if similarity_threshold is not None:
            # validate threshold
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
    
    # override pipeline if threshold provided
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
            is_relevant=result["confidence"]["is_relevant"],
            level=result["confidence"]["level"],
            explanation=result["confidence"]["explanation"]
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

# file upload route
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    clear_index: bool = Form(False)
):
    logger.info(f"Received file upload: {file.filename}")
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Generate a unique filename
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_dir / unique_filename
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file to {file_path}")
        
        # Initialize counters
        documents_processed = 0
        chunks_created = 0
        chunks_indexed = 0
        
        # Process based on file type
        if file_extension == '.pdf':
            # Process PDF
            logger.info(f"Processing PDF file: {file_path}")
            
            # Extract text using OCR if needed
            doc_text = PDFProcessor.extract_text_from_pdf(str(file_path))
            if not doc_text:
                return JSONResponse(
                    status_code=400,
                    content={
                        "filename": original_filename,
                        "status": "error",
                        "documents_processed": 0,
                        "chunks_created": 0,
                        "chunks_indexed": 0,
                        "message": "Failed to extract text from PDF"
                    }
                )
            
            # Create document
            documents = [{
                "title": Path(original_filename).stem,
                "text": doc_text,
                "source": original_filename,
                "key": f"pdf_{Path(original_filename).stem}"
            }]
            documents_processed = 1
            
        elif file_extension == '.csv':
            # Process CSV
            logger.info(f"Processing CSV file: {file_path}")
            documents = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                rows = list(csv_reader)
                
                # Check required fields
                first_row = rows[0] if rows else {}
                has_title = 'title' in first_row
                has_contents = 'contents' in first_row or 'text' in first_row
                
                if not (has_title and has_contents):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "filename": original_filename,
                            "status": "error",
                            "documents_processed": 0,
                            "chunks_created": 0,
                            "chunks_indexed": 0,
                            "message": "CSV must have 'title' and 'contents' or 'text' columns"
                        }
                    )
                
                # Process rows
                for row_idx, row in enumerate(rows):
                    title = row.get('title', f"Document {row_idx+1}")
                    # Try both 'contents' and 'text' fields
                    content = row.get('contents', row.get('text', ''))
                    source = row.get('link', row.get('source', original_filename))
                    key = row.get('key', f"csv_{row_idx}_{Path(original_filename).stem}")
                    
                    if content:
                        documents.append({
                            "title": title,
                            "text": content,
                            "source": source,
                            "key": key
                        })
                
                documents_processed = len(documents)
                
        elif file_extension == '.json':
            # Process JSON
            logger.info(f"Processing JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both arrays and single objects
            if not isinstance(data, list):
                data = [data]
            
            documents = []
            for doc_idx, item in enumerate(data):
                # Check for required fields with different potential names
                title = item.get('title', item.get('Title', f"Document {doc_idx+1}"))
                content = item.get('text', item.get('Text', item.get('contents', item.get('content', ''))))
                source = item.get('source', item.get('Source', item.get('link', item.get('url', original_filename))))
                key = item.get('key', item.get('Key', item.get('id', f"json_{doc_idx}_{Path(original_filename).stem}")))
                
                if content:
                    documents.append({
                        "title": title,
                        "text": content,
                        "source": source,
                        "key": key
                    })
            
            documents_processed = len(documents)
            
        else:
            # Unsupported file type
            return JSONResponse(
                status_code=400,
                content={
                    "filename": original_filename,
                    "status": "error",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "message": f"Unsupported file type: {file_extension}"
                }
            )
        
        # Check if any documents were processed
        if not documents:
            return JSONResponse(
                status_code=400,
                content={
                    "filename": original_filename,
                    "status": "error",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "message": "No valid documents found in the file"
                }
            )
        
        # Chunk documents
        chunker = TextChunker(max_tokens=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(documents)
        chunks_created = len(chunks)
        
        if not chunks:
            return JSONResponse(
                status_code=400,
                content={
                    "filename": original_filename,
                    "status": "error",
                    "documents_processed": documents_processed,
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "message": "Failed to create chunks from documents"
                }
            )
        
        # Create embeddings and upload to Pinecone
        embedding_service = EmbeddingService()
        
        # Clear index if requested
        if clear_index:
            logger.info("Clearing Pinecone index before upload")
            embedding_service.clear_pinecone_index()
        
        # Process and upload chunks
        embedded_chunks = embedding_service.create_embeddings(chunks)
        embedding_service.upload_to_pinecone(embedded_chunks)
        chunks_indexed = len(embedded_chunks)
        
        # Return success response
        return {
            "filename": original_filename,
            "status": "success",
            "documents_processed": documents_processed,
            "chunks_created": chunks_created,
            "chunks_indexed": chunks_indexed,
            "message": f"Successfully processed {documents_processed} documents, created {chunks_created} chunks, and indexed {chunks_indexed} chunks"
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up the uploaded file
        if file_path.exists():
            file_path.unlink()
            
        return JSONResponse(
            status_code=500,
            content={
                "filename": original_filename,
                "status": "error",
                "documents_processed": 0,
                "chunks_created": 0,
                "chunks_indexed": 0,
                "message": f"Error processing file: {str(e)}"
            }
        )

# entrypoint
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

# run main
if __name__ == "__main__":
    main()