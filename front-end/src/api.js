// api.js
const API_BASE_URL = 'http://localhost:3001';

/**
 * Send a query to the RAG system
 * @param {string} query - The query text
 * @param {boolean} includeCitations - Whether to include citations in the response
 * @param {number} similarityThreshold - Optional similarity threshold override
 * @returns {Promise<Object>} - The query response
 */
export async function sendQuery(query, includeCitations = false, similarityThreshold = null) {
  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        with_citations: includeCitations,
        similarity_threshold: similarityThreshold
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error querying RAG system:', error);
    throw error;
  }
}

/**
 * Get the current similarity threshold setting
 * @returns {Promise<number>} - The current threshold
 */
export async function getSimilarityThreshold() {
  try {
    const response = await fetch(`${API_BASE_URL}/settings/similarity-threshold`);
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    const data = await response.json();
    return data.similarity_threshold;
  } catch (error) {
    console.error('Error fetching similarity threshold:', error);
    throw error;
  }
}

/**
 * Update the similarity threshold setting
 * @param {number} threshold - The new threshold value (0-1)
 * @returns {Promise<Object>} - The update response
 */
export async function updateSimilarityThreshold(threshold) {
  try {
    const response = await fetch(`${API_BASE_URL}/settings/similarity-threshold?threshold=${threshold}`, {
      method: 'PUT',
    });
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error updating similarity threshold:', error);
    throw error;
  }
}

/**
 * Check API health
 * @returns {Promise<Object>} - Health check response
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}