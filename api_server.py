from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
import threading

from rag_pipeline import create_rag_pipeline_from_env
from text_similarity import TextSimilarityCalculator

load_dotenv()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    with_citations: bool = True
    include_similarity: bool = True


pipeline = create_rag_pipeline_from_env()
openai_api_key = os.getenv("OPENAI_API_KEY")
similarity_calculator = TextSimilarityCalculator(openai_api_key=openai_api_key)

@app.post("/api/query")
async def process_query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        
        result = pipeline.process_query(request.query, with_citations=request.with_citations)
        
        
        if request.include_similarity and similarity_calculator:
            similarity_result = similarity_calculator.process_results_with_similarity(
                request.query, 
                result["retrieved_docs"]
            )
            result["retrieved_docs"] = similarity_result["results"]
            result["confidence"] = similarity_result["confidence"]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "API server is running"}


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()