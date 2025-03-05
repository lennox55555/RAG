import os
from typing import List, Dict, Any, Optional
import openai

class ResponseGenerator:
    """
    Class for generating responses based on retrieved documents using OpenAI's GPT models.
    """
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        """
        Initialize the ResponseGenerator.
        """
        # Initialize OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response based on the query and context.
        """
        prompt = f"""
You are a helpful assistant that answers questions based on the provided context.
Your answers should be informative, accurate, and based exclusively on the information in the context.
If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer that question." Do not make up information.

Context:
{context}

Question: {query}

Answer:
"""
        try:
            # Using the new OpenAI client
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def generate_response_with_citations(self, query: str, context: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response with citations based on the query and context.
        """
        sources = {}
        for i, doc in enumerate(results):
            sources[i+1] = {"title": doc["doc_title"], "source": doc["source"]}
        
        prompt = f"""
You are a helpful assistant that answers questions based on the provided context.
Your answers should be informative, accurate, and based exclusively on the information in the context.
If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer that question." Do not make up information.

When you use information from a specific document, cite the source using the document number in square brackets like this: [X].
Include multiple citations if information comes from multiple documents.

Context:
{context}

Question: {query}

Answer (with citations):
"""
        try:
            # Using the new OpenAI client
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            response_text = response.choices[0].message.content.strip()
            return {"response": response_text, "citations": sources}
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {"response": "I apologize, but I encountered an error while generating a response. Please try again.", "citations": {}}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from retriever import DocumentRetriever
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "your-default-pinecone-key")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
    
    retriever = DocumentRetriever(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key
    )
    
    generator = ResponseGenerator(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo"  # Use gpt-3.5-turbo for testing
    )
    
    query = "What happened to JFK?"
    results, formatted_context = retriever.retrieve_and_format(query)
    
    response = generator.generate_response(query, formatted_context)
    print(f"Query: {query}\nResponse:\n{response}")
    
    response_with_citations = generator.generate_response_with_citations(query, formatted_context, results)
    print(f"Response with citations:\n{response_with_citations['response']}")