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
 * Upload a file to the RAG system
 * @param {File} file - The file to upload
 * @param {Object} options - Upload options
 * @param {number} options.chunkSize - The chunk size for document splitting
 * @param {number} options.chunkOverlap - The overlap between chunks
 * @param {boolean} options.clearIndex - Whether to clear the index before upload
 * @returns {Promise<Object>} - The upload response with task ID
 */
export async function uploadFile(file, options = {}) {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunk_size', options.chunkSize || 1000);
    formData.append('chunk_overlap', options.chunkOverlap || 100);
    formData.append('clear_index', options.clearIndex || false);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `Upload failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
}

/**
 * Get the status of a file upload task
 * @param {string} taskId - The task ID to check
 * @returns {Promise<Object>} - The task status response
 */
export async function getUploadStatus(taskId) {
  try {
    const response = await fetch(`${API_BASE_URL}/upload/status/${taskId}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Upload task not found');
      }
      throw new Error(`Failed to get upload status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching upload status:', error);
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