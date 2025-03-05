import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import openai
from scipy.spatial.distance import cosine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSimilarityCalculator:

    # class for calculating similarity between text embeddings and presenting similarity metrics.

    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-ada-002"):
        
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.embedding_model = embedding_model
    
    def get_embedding(self, text: str) -> List[float]:
 
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback (not ideal but prevents pipeline failure)
            return [0.0] * 1536  
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:

        # Convert to numpy arrays for calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(vec1, vec2)
        
        return similarity
    
    def compare_query_to_results(self, 
                                query: str, 
                                results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compare query to each result and calculate similarity metrics.
        
        Args:
            query: User query
            results: List of result dictionaries from retriever
            
        Returns:
            List of results with similarity metrics added
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarity for each result
        for result in results:
            # Get text embedding if not already present
            if 'embedding' not in result:
                # If we don't have the embedding, we can try to get it from the text
                if 'text' in result:
                    text_embedding = self.get_embedding(result['text'])
                    similarity = self.calculate_similarity(query_embedding, text_embedding)
                else:
                    # If we don't have text either, we can't calculate similarity
                    similarity = 0.0
            else:
                similarity = self.calculate_similarity(query_embedding, result['embedding'])
            
            # Add similarity to result
            result['similarity'] = similarity
            
            # Add similarity category for UI display
            if similarity >= 0.9:
                result['similarity_category'] = 'Very High'
            elif similarity >= 0.75:
                result['similarity_category'] = 'High'
            elif similarity >= 0.6:
                result['similarity_category'] = 'Moderate'
            elif similarity >= 0.4:
                result['similarity_category'] = 'Low'
            else:
                result['similarity_category'] = 'Very Low'
        
        # Sort results by similarity (highest first)
        results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        return results

    def get_overall_confidence(self, similarities: List[float]) -> Dict[str, Any]:
        """
        Calculate overall confidence based on similarities of top results.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            Dictionary with confidence metrics
        """
        if not similarities:
            return {
                'score': 0.0,
                'level': 'Unknown',
                'explanation': 'No relevant documents found'
            }
        
        # Calculate weighted average of similarities
        # Put more weight on the top results
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        # Ensure we don't go out of bounds
        weights = weights[:len(similarities)]
        similarities = similarities[:len(weights)]
        
        # Calculate weighted average
        weighted_sum = sum(w * s for w, s in zip(weights, similarities))
        weight_sum = sum(weights[:len(similarities)])
        confidence_score = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Determine confidence level
        if confidence_score >= 0.85:
            level = 'Very High'
            explanation = 'The system found very relevant documents that closely match your query.'
        elif confidence_score >= 0.7:
            level = 'High'
            explanation = 'The system found relevant documents that match your query well.'
        elif confidence_score >= 0.55:
            level = 'Moderate'
            explanation = 'The system found somewhat relevant documents related to your query.'
        elif confidence_score >= 0.4:
            level = 'Low'
            explanation = 'The system found only marginally relevant documents for your query.'
        else:
            level = 'Very Low'
            explanation = 'The system could not find closely relevant documents matching your query.'
        
        return {
            'score': confidence_score,
            'level': level,
            'explanation': explanation
        }

    def process_results_with_similarity(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    
        # Add similarity metrics to results
        results_with_similarity = self.compare_query_to_results(query, results)
        
        # Get similarities for confidence calculation
        similarities = [r.get('similarity', 0) for r in results_with_similarity]
        
        # Calculate overall confidence
        confidence = self.get_overall_confidence(similarities)
        
        # Prepare response for frontend
        return {
            'query': query,
            'results': results_with_similarity,
            'confidence': confidence,
            'num_results': len(results_with_similarity)
        }


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    from retriever import DocumentRetriever
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "your-default-pinecone-key")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
    
    # Create retriever
    retriever = DocumentRetriever(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key
    )
    
    # Create similarity calculator
    similarity_calculator = TextSimilarityCalculator(
        openai_api_key=openai_api_key
    )
    
    # Test query
    query = "What happened to JFK?"
    results, _ = retriever.retrieve_and_format(query)
    
    # Process results with similarity
    processed_results = similarity_calculator.process_results_with_similarity(query, results)
    
    
    print(f"Query: {processed_results['query']}")
    print(f"Confidence: {processed_results['confidence']['level']} ({processed_results['confidence']['score']:.2f})")
    print(f"Explanation: {processed_results['confidence']['explanation']}")
    print(f"\nTop results:")
    
    for i, result in enumerate(processed_results['results'][:3], 1):
        print(f"{i}. {result['doc_title']}")
        print(f"   Similarity: {result['similarity']:.4f} ({result['similarity_category']})")
        print(f"   Text: {result['text'][:100]}...")
        print()