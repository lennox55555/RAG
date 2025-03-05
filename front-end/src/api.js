import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const fetchAnswer = async (query, withCitations = true, withSimilarity = false) => {
  try {
    const response = await axios.post(`${API_URL}/query`, {
      query,
      with_citations: withCitations,
      include_similarity: withSimilarity
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching answer:', error);
    throw error;
  }
};