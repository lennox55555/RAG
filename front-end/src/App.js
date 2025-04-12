import React, { useState } from 'react';
import { Search, Book, Loader, ArrowRight, FileText, BarChart2, Upload } from 'lucide-react';
import { sendQuery } from './api';
import FileUpload from './FileUpload';
import './FileUpload.css';

function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [showCitations, setShowCitations] = useState(true);
  const [showSimilarity, setShowSimilarity] = useState(true);
  const [activeTab, setActiveTab] = useState('search'); // 'search' or 'upload'

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
    <div className="mff-app">
      <header className="mff-header">
        <div className="mff-container">
          <div className="mff-logo">
            <img src="/mff-logo.png" alt="Logo" className="mff-logo-image" />
            <div className="mff-title-container">
              <h1 className="mff-title">MARY FERRELL FOUNDATION</h1>
              <h2 className="mff-subtitle">unredacting history</h2>
            </div>
          </div>
        </div>
      </header>

      <nav className="mff-nav">
        <div className="mff-container">
          <ul className="mff-nav-list">
            <li className="mff-nav-item mff-nav-active">MFF ASSISTANT</li>
            <li className="mff-nav-item">ARCHIVE</li>
            <li className="mff-nav-item">RESOURCES</li>
            <li className="mff-nav-item">ABOUT</li>
          </ul>
        </div>
      </nav>

      <main className="mff-container mff-main">
        <div className="mff-page-title">
          <h2>MFF Research Assistant</h2>
          <p>Search and manage documents in the Mary Ferrell Foundation archive</p>
        </div>

        <div className="mff-tabs">
          <button 
            className={`mff-tab ${activeTab === 'search' ? 'mff-tab-active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            <Search size={16} className="mff-tab-icon" />
            Search
          </button>
          <button 
            className={`mff-tab ${activeTab === 'upload' ? 'mff-tab-active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={16} className="mff-tab-icon" />
            Upload
          </button>
        </div>

        {activeTab === 'search' ? (
          <div className="mff-search-section">
            <form onSubmit={handleSubmit} className="mff-search-form">
              <div className="mff-search-container">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="mff-search-input"
                  placeholder="Ask a question about the JFK assassination, Civil Rights, or other historical events..."
                  required
                />
                <button
                  type="submit"
                  disabled={isLoading}
                  className="mff-search-button"
                >
                  {isLoading ? <Loader size={16} className="mff-spinner" /> : 'Search'}
                </button>
              </div>
              <div className="mff-checkbox-container">
                <div className="mff-checkbox-group">
                  <input
                    type="checkbox"
                    id="citations"
                    checked={showCitations}
                    onChange={() => setShowCitations(!showCitations)}
                    className="mff-checkbox"
                  />
                  <label htmlFor="citations" className="mff-checkbox-label">
                    Include citations
                  </label>
                </div>
                
                <div className="mff-checkbox-group">
                  <input
                    type="checkbox"
                    id="similarity"
                    checked={showSimilarity}
                    onChange={() => setShowSimilarity(!showSimilarity)}
                    className="mff-checkbox"
                  />
                  <label htmlFor="similarity" className="mff-checkbox-label">
                    Show similarity metrics
                  </label>
                </div>
              </div>
            </form>

            {isLoading && (
              <div className="mff-loader">
                <Loader className="mff-spinner" size={32} />
                <span className="mff-loader-text">Searching documents...</span>
              </div>
            )}

            {error && (
              <div className="mff-error">
                <p className="mff-error-text">
                  {error}
                </p>
              </div>
            )}

            {response && !isLoading && (
              <div className="mff-response">
                <h3 className="mff-response-title">Response</h3>
                <div 
                  className="mff-response-content"
                  dangerouslySetInnerHTML={{ 
                    __html: formatResponseWithCitations(response.response) 
                  }}
                />
                
                {/* Confidence section removed as requested */}
                
                {/* Source documents with similarity */}
                {showSimilarity && response.retrieved_docs && response.retrieved_docs.length > 0 && (
                  <div className="mff-sources">
                    <h4 className="mff-section-title">Source Documents</h4>
                    <div className="mff-sources-list">
                      {response.retrieved_docs.map((doc, index) => (
                        <div key={index} className="mff-source-item">
                          <div className="mff-source-header">
                            <div className="mff-source-title-info">
                              <h5 className="mff-source-title">{doc.doc_title}</h5>
                              {doc.page_num > 0 && (
                                <span className="mff-page-info">
                                  Page {doc.page_num}{doc.total_pages > 0 ? ` of ${doc.total_pages}` : ''}
                                </span>
                              )}
                            </div>
                            {doc.similarity && (
                              <div className="mff-similarity">
                                <div className="mff-similarity-indicator" 
                                  style={{ backgroundColor: getSimilarityColor(doc.similarity_category) }}>
                                </div>
                                <span className="mff-similarity-score">
                                  {Math.round(doc.similarity * 100)}% match
                                </span>
                              </div>
                            )}
                          </div>
                          <p className="mff-source-preview">
                            {doc.text && doc.text.length > 150 ? `${doc.text.substring(0, 150)}...` : doc.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Citations */}
                {response.citations && Object.keys(response.citations).length > 0 && (
                  <div className="mff-citations">
                    <h4 className="mff-section-title">Sources</h4>
                    <ul className="mff-citations-list">
                      {Object.entries(response.citations).map(([key, source]) => (
                        <li key={key} className="mff-citation-item">
                          <span className="mff-citation-number">[{key}]</span>
                          <span className="mff-citation-text">
                            <span className="mff-citation-title">{source.title}</span>
                            {source.page_num > 0 && <span className="mff-citation-page"> (Page {source.page_num})</span>}
                            {source.source !== source.title && ` - ${source.source}`}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="mff-metadata">
                  <FileText size={16} className="mff-metadata-icon" />
                  <span>
                    {response.num_docs_retrieved} documents retrieved
                  </span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <FileUpload />
        )}
      </main>

      <footer className="mff-footer">
        <div className="mff-container">
          <p className="mff-footer-text">
            © {new Date().getFullYear()} Mary Ferrell Foundation - Research Assistant
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;