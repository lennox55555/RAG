import axios from 'axios';

// Point to your deployed Streamlit app URL
const STREAMLIT_URL = 'https://cusfur3mwz8svmncsjjvvd.streamlit.app/';

export const fetchAnswer = async (query, withCitations = true, withSimilarity = false) => {
  try {
    // Create URL with query parameters for Streamlit
    const searchParams = new URLSearchParams({
      query: query,
      show_citations: withCitations,
      show_similarity: withSimilarity,
      submit: true,
      format: 'json'  // Request JSON format if supported
    });
    
    const streamlitUrl = `${STREAMLIT_URL}?${searchParams.toString()}`;
    
    return {
      response: `Your query has been sent to the RAG system. <a href="${streamlitUrl}" target="_blank" rel="noopener noreferrer">View Results in Streamlit App</a>`,
      retrieved_docs: [],
      num_docs_retrieved: 0,
      confidence: {
        level: 'Redirected',
        score: 1.0,
        explanation: 'Click the link above to see your results in the Streamlit application.'
      }
    };
    

  } catch (error) {
    console.error('Error connecting to Streamlit:', error);
    throw error;
  }
};