import os
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from retriever import DocumentRetriever
from generator import ResponseGenerator

class RAGPipeline:
    """
    Class that combines retrieval and generation for a complete RAG pipeline.
    """
    def __init__(self,
                 pinecone_api_key: str,
                 pinecone_index_name: str,
                 openai_api_key: str,
                 pinecone_namespace: str = "documents",
                 retriever_top_k: int = 5,
                 generator_model: str = "gpt-4",
                 generator_temperature: float = 0.7,
                 generator_max_tokens: int = 500):
        """
        Initialize the RAG pipeline.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_index_name: Name of the Pinecone index
            openai_api_key: OpenAI API key
            pinecone_namespace: Namespace in the Pinecone index
            retriever_top_k: Number of top results to retrieve
            generator_model: Name of the GPT model to use
            generator_temperature: Temperature for response generation
            generator_max_tokens: Maximum number of tokens in the response
        """
        # Initialize retriever
        self.retriever = DocumentRetriever(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            pinecone_namespace=pinecone_namespace,
            openai_api_key=openai_api_key,
            top_k=retriever_top_k
        )
        
        # Initialize generator
        self.generator = ResponseGenerator(
            openai_api_key=openai_api_key,
            model=generator_model,
            temperature=generator_temperature,
            max_tokens=generator_max_tokens
        )
    
    def process_query(self, query: str, with_citations: bool = False) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            with_citations: Whether to include citations in the response
            
        Returns:
            Dictionary with response and metadata
        """
        # Step 1: Retrieve relevant documents
        results, formatted_context = self.retriever.retrieve_and_format(query)
        
        # Step 2: Generate response
        if with_citations:
            response_data = self.generator.generate_response_with_citations(query, formatted_context, results)
            response_text = response_data["response"]
            citations = response_data["citations"]
        else:
            response_text = self.generator.generate_response(query, formatted_context)
            citations = {}
        
        # Step 3: Prepare response
        return {
            "query": query,
            "response": response_text,
            "citations": citations,
            "retrieved_docs": results,
            "num_docs_retrieved": len(results)
        }


def create_rag_pipeline_from_env() -> RAGPipeline:
    """
    Create a RAG pipeline using environment variables.
    
    Returns:
        RAG pipeline instance
    """
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
    
    return RAGPipeline(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key,
        generator_model=os.getenv("GENERATOR_MODEL", "gpt-3.5-turbo")
    )


if __name__ == "__main__":
    try:
        pipeline = create_rag_pipeline_from_env()
        
        query = "What happened to JFK?"
        
        result = pipeline.process_query(query, with_citations=True)
        
        print(f"Query: {result['query']}")
        print(f"Retrieved {result['num_docs_retrieved']} documents")
        print(f"\nResponse:\n{result['response']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")