from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Callable
import os
import uuid
import uvicorn
import traceback
import logging
import tempfile
import json
import csv
import shutil
import zipfile
import io
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

# Task manager for background processing
class TaskManager:
    tasks = {}

    @classmethod
    def create_task(cls, task_id=None):
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        cls.tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "message": "Task created and waiting to start",
            "is_complete": False,
            "error": None
        }
        return task_id
    
    @classmethod
    def update_task(cls, task_id, **kwargs):
        if task_id in cls.tasks:
            cls.tasks[task_id].update(kwargs)
    
    @classmethod
    def get_task(cls, task_id):
        return cls.tasks.get(task_id)
    
    @classmethod
    def clean_old_tasks(cls, max_tasks=100):
        """Remove completed tasks if there are too many"""
        # Get all completed tasks
        completed_tasks = [tid for tid, task in cls.tasks.items() 
                         if task.get("is_complete", False)]
        
        # If we have more than max_tasks, remove the oldest ones
        if len(cls.tasks) > max_tasks:
            # Sort by completion time if available, otherwise just take the first ones
            for task_id in completed_tasks[:(len(cls.tasks) - max_tasks)]:
                cls.tasks.pop(task_id, None)

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

# upload response models
class UploadResponse(BaseModel):
    filename: str
    status: str
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    message: str
    task_id: Optional[str] = None

class UploadStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float  # 0-100
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    message: str
    is_complete: bool

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

# Get upload status
@app.get("/upload/status/{task_id}", response_model=UploadStatusResponse)
async def get_upload_status(task_id: str):
    task_info = TaskManager.get_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
    
    # Ensure all required fields are present
    default_fields = {
        "status": "processing",
        "progress": 0.0,
        "documents_processed": 0,
        "chunks_created": 0,
        "chunks_indexed": 0,
        "message": "Processing...",
        "is_complete": False
    }
    
    # Merge with task_info, using defaults for missing fields
    result = {**default_fields, **task_info, "task_id": task_id}
    
    return result

# Process file in background
async def process_file_in_background(
    task_id: str,
    file_path: Path,
    original_filename: str,
    chunk_size: int,
    chunk_overlap: int,
    clear_index: bool
):
    try:
        # Update task status
        TaskManager.update_task(
            task_id,
            status="processing",
            message=f"Processing {original_filename}"
        )
        
        # Initialize counters
        documents_processed = 0
        chunks_created = 0
        chunks_indexed = 0
        all_chunks = []
        all_documents = []
        
        # Process based on file type
        file_extension = file_path.suffix.lower()
        
        # Process ZIP file - extract and process each file
        if file_extension == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Count total files for progress tracking
                total_files = len([f for f in zip_ref.namelist() if not f.endswith('/')])
                files_processed = 0
                
                # Create temp directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract all files
                    zip_ref.extractall(temp_dir)
                    temp_path = Path(temp_dir)
                    
                    # Process each file
                    for zip_file in zip_ref.namelist():
                        if zip_file.endswith('/'):  # Skip directories
                            continue
                        
                        extracted_path = temp_path / zip_file
                        if not extracted_path.exists():
                            continue
                            
                        zip_file_extension = extracted_path.suffix.lower()
                        
                        # Process based on file type
                        if zip_file_extension == '.pdf':
                            # Process PDF
                            logger.info(f"Processing PDF from ZIP: {zip_file} ({files_processed+1}/{total_files})")
                            
                            # Base progress for this file in the zip (0-45% range spread across files)
                            base_progress = (files_processed / total_files) * 45
                            # Update with initial status
                            TaskManager.update_task(
                                task_id,
                                message=f"Processing PDF in ZIP archive: {zip_file} ({files_processed+1}/{total_files})",
                                progress=base_progress
                            )
                            
                            # Define a progress callback for PDF extraction within ZIP
                            def zip_pdf_progress_callback(stage, current, total):
                                if total == 0:
                                    return  # Avoid division by zero
                                    
                                # Calculate progress within this file (up to 3% per file)
                                if stage == 'extract':
                                    # Text extraction phase 
                                    file_progress_pct = (current / total) * 1.5
                                    message = f"Extracting text from {Path(zip_file).name}: page {current} of {total}"
                                elif stage == 'ocr':
                                    # OCR phase 
                                    file_progress_pct = 1.5 + (current / total) * 1.5
                                    message = f"Performing OCR on {Path(zip_file).name}: page {current} of {total}"
                                else:
                                    return
                                
                                # Update task with progress within this file plus base progress
                                TaskManager.update_task(
                                    task_id,
                                    message=message,
                                    progress=base_progress + file_progress_pct
                                )
                            
                            # Extract text with page information and progress tracking
                            pdf_result = PDFProcessor.extract_text_from_pdf(
                                str(extracted_path), 
                                progress_callback=zip_pdf_progress_callback
                            )
                            
                            if pdf_result["full_text"]:
                                all_documents.append({
                                    "title": Path(zip_file).stem,
                                    "text": pdf_result["full_text"],
                                    "source": f"{original_filename}:{zip_file}",
                                    "key": f"pdf_{Path(zip_file).stem}",
                                    "pages": pdf_result["pages_text"],
                                    "total_pages": pdf_result["total_pages"]
                                })
                                documents_processed += 1
                                
                                # Update completion status for this file
                                TaskManager.update_task(
                                    task_id,
                                    message=f"Finished processing PDF {files_processed+1}/{total_files}: {Path(zip_file).name}",
                                    progress=base_progress + 3  # Each file gets about 3% progress
                                )
                        
                        # Could add handlers for CSV and JSON here if needed
                            
                        files_processed += 1
                        TaskManager.update_task(
                            task_id,
                            documents_processed=documents_processed,
                            progress=(files_processed / total_files) * 50
                        )
        
        # Process PDF file directly
        elif file_extension == '.pdf':
            # Process PDF
            logger.info(f"Processing PDF file: {file_path}")
            TaskManager.update_task(
                task_id, 
                message=f"Preparing to extract text from PDF",
                progress=5
            )
            
            # Define a progress callback function for PDF processing
            def pdf_progress_callback(stage, current, total):
                if total == 0:
                    return  # Avoid division by zero
                    
                # Calculate progress percentage (5-25% of overall process)
                if stage == 'extract':
                    # Text extraction phase (5-15%)
                    progress_pct = 5 + (current / total) * 10
                    message = f"Extracting text from PDF: page {current} of {total}"
                elif stage == 'ocr':
                    # OCR phase (15-25%)
                    progress_pct = 15 + (current / total) * 10
                    message = f"Performing OCR on PDF: page {current} of {total}"
                else:
                    return
                
                # Update task status
                TaskManager.update_task(
                    task_id,
                    message=message,
                    progress=progress_pct
                )
            
            # Extract text using OCR if needed, with page information and progress tracking
            pdf_result = PDFProcessor.extract_text_from_pdf(
                str(file_path), 
                progress_callback=pdf_progress_callback
            )
            
            if pdf_result["full_text"]:
                all_documents.append({
                    "title": Path(original_filename).stem,
                    "text": pdf_result["full_text"],
                    "source": original_filename,
                    "key": f"pdf_{Path(original_filename).stem}",
                    "pages": pdf_result["pages_text"],
                    "total_pages": pdf_result["total_pages"]
                })
                documents_processed = 1
                
                TaskManager.update_task(
                    task_id,
                    documents_processed=documents_processed,
                    message=f"Finished extracting text: {pdf_result['total_pages']} pages processed",
                    progress=25
                )
        
        # Process CSV file
        elif file_extension == '.csv':
            # Process CSV
            logger.info(f"Processing CSV file: {file_path}")
            TaskManager.update_task(
                task_id, 
                message=f"Processing CSV file",
                progress=10
            )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                rows = list(csv_reader)
                
                # Check required fields
                first_row = rows[0] if rows else {}
                has_title = 'title' in first_row
                has_contents = 'contents' in first_row or 'text' in first_row
                
                if not (has_title and has_contents):
                    TaskManager.update_task(
                        task_id,
                        status="error",
                        message="CSV must have 'title' and 'contents' or 'text' columns",
                        is_complete=True
                    )
                    return
                
                # Process rows
                total_rows = len(rows)
                for row_idx, row in enumerate(rows):
                    title = row.get('title', f"Document {row_idx+1}")
                    content = row.get('contents', row.get('text', ''))
                    source = row.get('link', row.get('source', original_filename))
                    key = row.get('key', f"csv_{row_idx}_{Path(original_filename).stem}")
                    
                    if content:
                        all_documents.append({
                            "title": title,
                            "text": content,
                            "source": source,
                            "key": key
                        })
                        documents_processed += 1
                    
                    # Update progress
                    if row_idx % max(1, total_rows // 10) == 0:
                        TaskManager.update_task(
                            task_id,
                            documents_processed=documents_processed,
                            progress=10 + (row_idx / total_rows) * 10
                        )
        
        # Process JSON file
        elif file_extension == '.json':
            # Process JSON
            logger.info(f"Processing JSON file: {file_path}")
            TaskManager.update_task(
                task_id, 
                message=f"Processing JSON file",
                progress=10
            )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both arrays and single objects
            if not isinstance(data, list):
                data = [data]
            
            total_items = len(data)
            for doc_idx, item in enumerate(data):
                # Check for required fields
                title = item.get('title', item.get('Title', f"Document {doc_idx+1}"))
                content = item.get('text', item.get('Text', item.get('contents', item.get('content', ''))))
                source = item.get('source', item.get('Source', item.get('link', item.get('url', original_filename))))
                key = item.get('key', item.get('Key', item.get('id', f"json_{doc_idx}_{Path(original_filename).stem}")))
                
                if content:
                    all_documents.append({
                        "title": title,
                        "text": content,
                        "source": source,
                        "key": key
                    })
                    documents_processed += 1
                
                # Update progress
                if doc_idx % max(1, total_items // 10) == 0:
                    TaskManager.update_task(
                        task_id,
                        documents_processed=documents_processed,
                        progress=10 + (doc_idx / total_items) * 10
                    )
        
        else:
            # Unsupported file type
            TaskManager.update_task(
                task_id,
                status="error",
                message=f"Unsupported file type: {file_extension}",
                is_complete=True
            )
            return
        
        # Check if any documents were processed
        if not all_documents:
            TaskManager.update_task(
                task_id,
                status="error",
                message="No valid documents found in the file",
                is_complete=True
            )
            return
        
        # Chunk documents
        TaskManager.update_task(
            task_id,
            status="chunking",
            message=f"Preparing to chunk {len(all_documents)} documents",
            progress=50
        )
        
        # Custom chunking with progress updates
        chunker = TextChunker(max_tokens=chunk_size, chunk_overlap=chunk_overlap)
        
        # Progress updates during chunking
        doc_count = len(all_documents)
        for i, doc in enumerate(all_documents):
            # Update progress (50-55% range)
            current_progress = 50 + ((i+1) / doc_count) * 5
            TaskManager.update_task(
                task_id,
                message=f"Chunking document {i+1}/{doc_count}: {doc.get('title', 'Untitled')}",
                progress=current_progress
            )
            
        # Actual chunking
        all_chunks = chunker.chunk_documents(all_documents)
        chunks_created = len(all_chunks)
        
        TaskManager.update_task(
            task_id,
            message=f"Created {chunks_created} chunks from {len(all_documents)} documents",
            progress=55,
            chunks_created=chunks_created
        )
        
        if not all_chunks:
            TaskManager.update_task(
                task_id,
                status="error",
                message="Failed to create chunks from documents",
                is_complete=True
            )
            return
        
        # Create embeddings and upload to Pinecone
        TaskManager.update_task(
            task_id,
            status="embedding",
            message=f"Preparing to create embeddings for {chunks_created} chunks",
            progress=56,
            chunks_created=chunks_created
        )
        
        embedding_service = EmbeddingService()
        
        # Clear index if requested
        if clear_index:
            logger.info("Clearing Pinecone index before upload")
            TaskManager.update_task(
                task_id,
                message="Clearing index before upload",
                progress=58
            )
            embedding_service.clear_pinecone_index()
        
        # Define a progress callback for embedding creation
        original_create_embeddings = embedding_service.create_embeddings
        
        def create_embeddings_with_progress(chunks, save_path=None):
            # Wrap the original method to add progress tracking
            
            # Custom progress tracking tqdm class
            class ProgressTracker:
                def __init__(self, total, desc="Creating embeddings"):
                    self.total = total
                    self.n = 0
                    self.desc = desc
                
                def update(self, n=1):
                    self.n += n
                    # Progress range: 60-80%
                    progress = 60 + min(1.0, self.n / self.total) * 20
                    TaskManager.update_task(
                        task_id,
                        message=f"{self.desc}: {self.n}/{self.total} chunks ({(self.n/self.total)*100:.1f}%)",
                        progress=progress
                    )
            
            # Create a custom progress tracker
            progress_tracker = ProgressTracker(len(chunks))
            
            # We'll need to modify embedding creation to use our tracker
            # Monkey patch tqdm
            original_tqdm = tqdm.tqdm
            tqdm.tqdm = lambda iterable, **kwargs: progress_tracker
            
            try:
                # Call original method with our tracker
                result = original_create_embeddings(chunks, save_path)
                return result
            finally:
                # Restore tqdm
                tqdm.tqdm = original_tqdm
        
        # Monkey patch the embedding service temporarily
        embedding_service.create_embeddings = create_embeddings_with_progress
        
        # Define a progress callback for vector upload
        original_upload_to_pinecone = embedding_service.upload_to_pinecone
        
        def upload_to_pinecone_with_progress(embedded_chunks):
            # Wrap the original method to add progress tracking
            
            # Custom progress tracking tqdm class
            class UploadTracker:
                def __init__(self, total, desc="Uploading to Pinecone"):
                    self.total = total
                    self.n = 0
                    self.desc = desc
                
                def update(self, n=1):
                    self.n += n
                    # Progress range: 80-95%
                    progress = 80 + min(1.0, self.n / self.total) * 15
                    TaskManager.update_task(
                        task_id,
                        message=f"{self.desc}: {self.n}/{self.total} vectors ({(self.n/self.total)*100:.1f}%)",
                        progress=progress
                    )
            
            # Create a custom progress tracker
            progress_tracker = UploadTracker(len(embedded_chunks))
            
            # Monkey patch tqdm
            original_tqdm = tqdm.tqdm
            tqdm.tqdm = lambda iterable, **kwargs: progress_tracker
            
            try:
                # Call original method with our tracker
                original_upload_to_pinecone(embedded_chunks)
            finally:
                # Restore tqdm
                tqdm.tqdm = original_tqdm
        
        # Monkey patch the upload method temporarily
        embedding_service.upload_to_pinecone = upload_to_pinecone_with_progress
        
        # Execute embedding and uploading with progress tracking
        try:
            # Create embeddings
            embedded_chunks = embedding_service.create_embeddings(all_chunks)
            
            # Upload to vector database
            embedding_service.upload_to_pinecone(embedded_chunks)
            chunks_indexed = len(embedded_chunks)
        finally:
            # Restore original methods
            embedding_service.create_embeddings = original_create_embeddings
            embedding_service.upload_to_pinecone = original_upload_to_pinecone
        
        # Complete the task
        TaskManager.update_task(
            task_id,
            status="complete",
            message=f"Successfully processed {documents_processed} documents, created {chunks_created} chunks, and indexed {chunks_indexed} chunks",
            progress=100,
            chunks_indexed=chunks_indexed,
            is_complete=True
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        
        TaskManager.update_task(
            task_id,
            status="error",
            message=f"Error processing file: {str(e)}",
            error=str(e),
            is_complete=True
        )
    finally:
        # Clean up the uploaded file
        if file_path.exists():
            file_path.unlink()
        
        # Clean old tasks
        TaskManager.clean_old_tasks()

# file upload route
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    clear_index: bool = Form(False)
):
    logger.info(f"Received file upload: {file.filename}")
    
    # Check file extension
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()
    
    # Verify supported file types (now including zip)
    if file_extension not in ['.pdf', '.csv', '.json', '.zip']:
        return JSONResponse(
            status_code=400,
            content={
                "filename": original_filename,
                "status": "error",
                "documents_processed": 0,
                "chunks_created": 0,
                "chunks_indexed": 0,
                "message": f"Unsupported file type: {file_extension} - Only PDF, CSV, JSON, and ZIP files are supported"
            }
        )
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Generate a unique task ID and filename
    task_id = str(uuid.uuid4())
    unique_filename = f"{task_id}{file_extension}"
    file_path = upload_dir / unique_filename
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file to {file_path}")
        
        # Initialize task
        TaskManager.create_task(task_id)
        
        # Schedule background processing
        background_tasks.add_task(
            process_file_in_background,
            task_id=task_id,
            file_path=file_path,
            original_filename=original_filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clear_index=clear_index
        )
        
        # Return immediate response with task ID
        return {
            "filename": original_filename,
            "status": "processing",
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "message": "File upload started. Processing in background.",
            "task_id": task_id
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