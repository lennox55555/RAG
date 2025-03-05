import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import openai
from scipy.spatial.distance import cosine
import logging

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
            return [0.0] * 1536  
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        similarity = 1 - cosine(vec1, vec2)
        
        return similarity
    
    def compare_query_to_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        query_embedding = self.get_embedding(query)
        
        for result in results:
            if 'embedding' not in result:
                if 'text' in result:
                    text_embedding = self.get_embedding(result['text'])
                    similarity = self.calculate_similarity(query_embedding, text_embedding)
                else:
                    similarity = 0.0
            else:
                similarity = self.calculate_similarity(query_embedding, result['embedding'])
            
            result['similarity'] = similarity
            
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
        
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        weights = weights[:len(similarities)]
        similarities = similarities[:len(weights)]
        
        weighted_sum = sum(w * s for w, s in zip(weights, similarities))
        weight_sum = sum(weights[:len(similarities)])
        confidence_score = weighted_sum / weight_sum if weight_sum > 0 else 0
        
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
    
        results_with_similarity = self.compare_query_to_results(query, results)
        
        similarities = [r.get('similarity', 0) for r in results_with_similarity]
        
        confidence = self.get_overall_confidence(similarities)
        
        return {
            'query': query,
            'results': results_with_similarity,
            'confidence': confidence,
            'num_results': len(results_with_similarity)
        }


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
    
    similarity_calculator = TextSimilarityCalculator(
        openai_api_key=openai_api_key
    )
    
    query = "What happened to JFK?"
    results, _ = retriever.retrieve_and_format(query)
    
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