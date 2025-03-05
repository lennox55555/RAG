import axios from 'axios';

// Point to your PHP proxy on your domain
const PROXY_URL = 'https://lennoxanderson.com/mffrag/rag-proxy.php';

export const fetchAnswer = async (query, withCitations = true, withSimilarity = false) => {
  try {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('with_citations', withCitations.toString());
    formData.append('include_similarity', withSimilarity.toString());
    
    const response = await axios.post(PROXY_URL, formData);
    
    return response.data;
  } catch (error) {
    console.error('Error querying API:', error);
    throw new Error('Could not process your query. Please try again later.');
  }
};