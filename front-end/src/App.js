import React, { useState } from 'react';
import { Search, Book, Loader, ArrowRight, FileText, BarChart2 } from 'lucide-react';
import { sendQuery } from './api';

function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [showCitations, setShowCitations] = useState(true);
  const [showSimilarity, setShowSimilarity] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    
    try {
      const data = await sendQuery(query.trim(), showCitations, showSimilarity);
      setResponse(data);
      console.log("API Response:", data); // Debug logging
    } catch (err) {
      console.error('Error querying API:', err);
      setError(err.message || 'An error occurred while processing your query');
    } finally {
      setIsLoading(false);
    }
  };

  // Format citation references in response text
  const formatResponseWithCitations = (text) => {
    if (!text) return '';
    
    // Replace citation references like [1] with clickable spans
    return text.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<span class="citation-ref" data-citation="${p1}">${match}</span>`;
    });
  };

  // Helper function to get color for similarity badges
  const getSimilarityColor = (category) => {
    switch (category) {
      case 'Very High': return '#10b981'; // Green
      case 'High': return '#3b82f6';      // Blue
      case 'Moderate': return '#f59e0b';  // Yellow
      case 'Low': return '#f97316';       // Orange
      case 'Very Low': return '#ef4444';  // Red
      default: return '#6b7280';          // Gray
    }
  };

  return (
    <div>
      <header>
        <div className="container">
          <div className="header-content">
            <Book size={24} />
            <h1 className="header-title">Document RAG System</h1>
          </div>
        </div>
      </header>

      <main className="container main-content">
        <div>
          <form onSubmit={handleSubmit} className="search-container">
            <div className="search-icon">
              <Search size={20} />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="search-input"
              placeholder="Ask a question about the documents..."
              required
            />
            <button
              type="submit"
              disabled={isLoading}
              className="search-button"
            >
              {isLoading ? <Loader size={16} className="loader-icon" /> : <ArrowRight size={16} />}
              <span style={{ marginLeft: '0.5rem' }}>Search</span>
            </button>
          </form>
          <div className="checkbox-container">
            <input
              type="checkbox"
              id="citations"
              checked={showCitations}
              onChange={() => setShowCitations(!showCitations)}
              className="checkbox-input"
            />
            <label htmlFor="citations" className="checkbox-label">
              Include citations
            </label>
            
            <input
              type="checkbox"
              id="similarity"
              checked={showSimilarity}
              onChange={() => setShowSimilarity(!showSimilarity)}
              className="checkbox-input"
              style={{ marginLeft: '1rem' }}
            />
            <label htmlFor="similarity" className="checkbox-label">
              <BarChart2 size={16} style={{ marginRight: '0.25rem', display: 'inline' }} />
              Show similarity metrics
            </label>
          </div>

          {isLoading && (
            <div className="loader">
              <Loader className="loader-icon" size={32} />
              <span className="loader-text">Searching documents...</span>
            </div>
          )}

          {error && (
            <div className="error-message">
              <p className="error-text">
                {error}
              </p>
            </div>
          )}

          {response && !isLoading && (
            <div className="response-container">
              <h2 className="response-title">Response</h2>
              <div 
                className="response-text"
                dangerouslySetInnerHTML={{ 
                  __html: formatResponseWithCitations(response.response) 
                }}
              />
              
              {/* New section for confidence display */}
              {showSimilarity && response.confidence && (
                <div style={{ marginTop: '1.5rem', borderTop: '1px solid #e5e7eb', paddingTop: '1rem' }}>
                  <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#4b5563', marginBottom: '0.5rem' }}>
                    Confidence
                  </h3>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span style={{ 
                      padding: '0.25rem 0.75rem', 
                      borderRadius: '9999px', 
                      fontSize: '0.75rem', 
                      fontWeight: '500',
                      backgroundColor: `${getSimilarityColor(response.confidence.level)}20`,
                      color: getSimilarityColor(response.confidence.level),
                    }}>
                      {response.confidence.level}
                    </span>
                    <span style={{ marginLeft: '0.5rem', fontSize: '0.875rem', color: '#4b5563' }}>
                      {Math.round(response.confidence.score * 100)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                    {response.confidence.explanation}
                  </p>
                </div>
              )}
              
              {/* New section for source documents with similarity */}
              {showSimilarity && response.retrieved_docs && response.retrieved_docs.length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                  <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#4b5563', marginBottom: '0.5rem' }}>
                    Source Documents
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {response.retrieved_docs.map((doc, index) => (
                      <div key={index} style={{ 
                        padding: '0.75rem', 
                        backgroundColor: '#f9fafb', 
                        borderRadius: '0.375rem',
                        border: '1px solid #e5e7eb'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <h4 style={{ fontWeight: '500', color: '#1f2937', margin: '0' }}>
                            {doc.doc_title}
                          </h4>
                          {doc.similarity && (
                            <div style={{ display: 'flex', alignItems: 'center' }}>
                              <div style={{ 
                                width: '0.5rem', 
                                height: '0.5rem', 
                                borderRadius: '9999px', 
                                backgroundColor: getSimilarityColor(doc.similarity_category), 
                                marginRight: '0.5rem' 
                              }}></div>
                              <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                                {Math.round(doc.similarity * 100)}% match
                              </span>
                            </div>
                          )}
                        </div>
                        <p style={{ 
                          fontSize: '0.875rem', 
                          color: '#6b7280', 
                          marginTop: '0.25rem',
                          marginBottom: '0'
                        }}>
                          {doc.text && doc.text.length > 150 ? `${doc.text.substring(0, 150)}...` : doc.text}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {response.citations && Object.keys(response.citations).length > 0 && (
                <div className="citations">
                  <h3 className="citations-title">Sources</h3>
                  <ul className="citation-list">
                    {Object.entries(response.citations).map(([key, source]) => (
                      <li key={key} className="citation-item">
                        <span className="citation-number">[{key}]</span>
                        <span className="citation-text">
                          <span className="citation-title">{source.title}</span>
                          {source.source !== source.title && ` - ${source.source}`}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="metadata">
                <FileText size={16} className="metadata-icon" />
                <span>
                  {response.num_docs_retrieved} documents retrieved
                </span>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer>
        <div className="container">
          <p className="footer-text">
            © 2023 Document RAG System
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;