import axios from 'axios';


const STREAMLIT_URL = 'https://cusfur3mwz8svmncsjjvvd.streamlit.app/';

const BASE_URL = 'https://lennoxanderson.com/mffrag';

export const fetchAnswer = async (query, withCitations = true, withSimilarity = false) => {
  try {
   
    const searchParams = new URLSearchParams({
      query: query,
      show_citations: withCitations,
      show_similarity: withSimilarity,
      submit: true,
      format: 'json' 
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