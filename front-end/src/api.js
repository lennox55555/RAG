import axios from 'axios';

// This should point to your deployed Streamlit app's API endpoint
const API_URL = 'https://cusfur3mwz8svmncsjjvvd.streamlit.app';

export const fetchAnswer = async (query, withCitations = true, withSimilarity = false) => {
  try {
    const response = await axios.post(`${API_URL}/api/query`, {
      query: query,
      with_citations: withCitations,
      include_similarity: withSimilarity
    });
    
    return response.data;
  } catch (error) {
    console.error('Error connecting to RAG API:', error);
    
    // Return a meaningful error message to display to the user
    if (error.response) {
      // The request was made and the server responded with an error status
      throw new Error(`Server error: ${error.response.data.detail || error.response.statusText}`);
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('Could not reach the RAG server. Please check your connection or try again later.');
    } else {
      // Something happened in setting up the request
      throw new Error(`Error: ${error.message}`);
    }
  }
};