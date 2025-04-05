import os
from dotenv import load_dotenv
import openai
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import time
import numpy as np
from scipy.spatial.distance import cosine

# load env vars
load_dotenv()

# setup logging
from logger_config import setup_logger
logger = setup_logger("rag_pipeline")

# init openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

# pinecone config
pinecone_api_key = os.getenv("PINECONE_API_KEY", "your_actual_api_key")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "documents")
pinecone_host = "https://mff-foxube2.svc.aped-4627-b74a.pinecone.io"

class RAGPipeline:
    def __init__(self):
        logger.info("RAGPipeline initialized.")
        self.host = pinecone_host
        self.api_key = pinecone_api_key
        self.namespace = pinecone_namespace
        self.similarity_threshold = 0.6
        self.headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-3.5-turbo"
        self.max_tokens = 500
        self.temperature = 0.7

    def embed_query(self, query: str) -> List[float]:
        # generate query embedding
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=query
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def retrieve_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        # get relevant docs from pinecone
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

    def _filter_documents_by_similarity(self, matches: List[Dict]) -> List[Dict]:
        # filter by similarity threshold
        filtered = []
        for match in matches:
            similarity_score = match.get("score", 0)
            # keep docs above threshold or at least one doc
            if similarity_score >= self.similarity_threshold or not filtered:
                filtered.append(match)
        return filtered

    def _format_context(self, documents: List[Dict]) -> str:
        # format docs for context
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # extract text from metadata
            doc_text = doc["metadata"].get("text", "")
            doc_title = doc["metadata"].get("doc_title", "Unknown")
            doc_source = doc["metadata"].get("source", "")
            
            # truncate long texts
            if len(doc_text) > 750:
                doc_text = doc_text[:750] + "..."
                
            context_parts.append(f"Document {i}:\nTitle: {doc_title}\nSource: {doc_source}\nContent: {doc_text}\n")
        
        return "\n".join(context_parts)

    def generate_response(self, query: str, documents: List[Dict]) -> str:
        # generate response using llm
        try:
            # format context
            context = self._format_context(documents)
            
            # exit if no relevant docs
            if context.startswith("No relevant information"):
                return "I don't have specific information about that in my knowledge base. Please ask about a topic covered in the documents I have access to."
            
            prompt = f"""You are a helpful research assistant for the Mary Ferrell Foundation, which focuses on historical documents related to the JFK assassination, civil rights, and other significant historical events.

Query: {query}

Context from documents:
{context}

Based only on the provided context, please give a comprehensive and accurate answer to the query. 
If the context doesn't contain relevant information, acknowledge this limitation.
Format your response in a scholarly tone appropriate for historical research. Cite your sources using numbers in square brackets [1] where appropriate."""

            response = openai.ChatCompletion.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def process_query(self, query: str, with_citations: bool = False) -> Dict[str, Any]:
        # main query processing pipeline
        try:
            # get query embedding
            query_embedding = self.embed_query(query)

            # retrieve docs
            matches = self.retrieve_documents(query_embedding)
            num_docs = len(matches)

            # prepare citations and docs
            citations = {}
            retrieved_docs = []
            
            if matches:
                filtered_matches = self._filter_documents_by_similarity(matches)
                
                for i, match in enumerate(filtered_matches):
                    similarity_score = match.get("score", 0)
                    
                    doc_text = match["metadata"].get("text", "")
                    doc_title = match["metadata"].get("doc_title", "Unknown")
                    doc_source = match["metadata"].get("source", "")
                    
                    # add to retrieved docs for frontend
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

            # generate response
            if retrieved_docs:
                response = self.generate_response(query, filtered_matches)
                confidence_level = self._get_confidence_level(matches[0]["score"] if matches else 0)
            else:
                response = "I don't have specific information about that in my knowledge base. Please ask about a topic covered in the documents I have access to."
                confidence_level = {"score": 0.0, "is_relevant": False, "level": "Very Low", "explanation": "No relevant documents found."}

            # build result
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
        # convert score to category
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
        # get confidence info from score
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
    # create pipeline from env vars
    threshold = os.getenv("SIMILARITY_THRESHOLD")
    pipeline = RAGPipeline()
    if threshold:
        try:
            pipeline.similarity_threshold = float(threshold)
        except ValueError:
            logger.warning(f"Invalid similarity threshold '{threshold}', using default")
    return pipeline

if __name__ == "__main__":
    # test
    pipeline = create_rag_pipeline_from_env()
    result = pipeline.process_query("Who is JFK and how did he die?")
    print(json.dumps(result, indent=2))