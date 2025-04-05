import React, { useState } from 'react';
import { Upload, CheckCircle, AlertCircle, FileText, Database, Loader, X } from 'lucide-react';
import { uploadFile } from './api';

function FileUpload() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [options, setOptions] = useState({
    chunkSize: 1000,
    chunkOverlap: 100,
    clearIndex: false
  });

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setUploadResult(null);
    }
  };

  const handleOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    setOptions({
      ...options,
      [name]: type === 'checkbox' ? checked : Number(value)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'csv', 'json'].includes(fileExtension)) {
      setError('Only PDF, CSV, and JSON files are supported');
      return;
    }

    setIsUploading(true);
    setError(null);
    
    try {
      const result = await uploadFile(file, options);
      setUploadResult(result);
      setFile(null);
      // Reset file input
      e.target.reset();
    } catch (err) {
      setError(err.message || 'An error occurred while uploading the file');
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setUploadResult(null);
    setError(null);
  };

  return (
    <div className="mff-file-upload">
      <h3 className="mff-section-title">Upload Documents</h3>
      <p className="mff-section-description">
        Upload PDF, CSV, or JSON files to add documents to the knowledge base
      </p>

      {uploadResult ? (
        <div className={`mff-upload-result ${uploadResult.status === 'success' ? 'mff-upload-success' : 'mff-upload-error'}`}>
          {uploadResult.status === 'success' ? (
            <CheckCircle className="mff-result-icon mff-success-icon" size={40} />
          ) : (
            <AlertCircle className="mff-result-icon mff-error-icon" size={40} />
          )}

          <h4 className="mff-result-title">
            {uploadResult.status === 'success' ? 'Upload Successful' : 'Upload Failed'}
          </h4>
          
          <p className="mff-result-message">{uploadResult.message}</p>
          
          {uploadResult.status === 'success' && (
            <div className="mff-upload-stats">
              <div className="mff-stat">
                <FileText size={16} />
                <span>Documents processed: {uploadResult.documents_processed}</span>
              </div>
              <div className="mff-stat">
                <Database size={16} />
                <span>Chunks created: {uploadResult.chunks_created}</span>
              </div>
              <div className="mff-stat">
                <CheckCircle size={16} />
                <span>Chunks indexed: {uploadResult.chunks_indexed}</span>
              </div>
            </div>
          )}
          
          <button className="mff-button" onClick={resetUpload}>
            Upload Another File
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="mff-upload-form">
          <div className="mff-file-input-container">
            <label className="mff-file-input-label">
              <input 
                type="file" 
                onChange={handleFileChange} 
                accept=".pdf,.csv,.json"
                className="mff-file-input"
              />
              <div className="mff-file-input-ui">
                <Upload size={24} className="mff-upload-icon" />
                <span className="mff-upload-text">
                  {file ? file.name : 'Choose a file or drag it here'}
                </span>
              </div>
            </label>
            {file && (
              <div className="mff-selected-file">
                <FileText size={16} className="mff-file-icon" />
                <span className="mff-file-name">{file.name}</span>
                <button 
                  type="button" 
                  className="mff-file-clear" 
                  onClick={() => setFile(null)}
                >
                  <X size={16} />
                </button>
              </div>
            )}
          </div>

          <div className="mff-upload-options">
            <h4 className="mff-options-title">Processing Options</h4>
            
            <div className="mff-option-group">
              <label className="mff-option-label">
                Chunk Size (tokens):
                <input
                  type="number"
                  name="chunkSize"
                  value={options.chunkSize}
                  onChange={handleOptionChange}
                  min="100"
                  max="8000"
                  className="mff-option-input"
                />
              </label>
              
              <label className="mff-option-label">
                Chunk Overlap (tokens):
                <input
                  type="number"
                  name="chunkOverlap"
                  value={options.chunkOverlap}
                  onChange={handleOptionChange}
                  min="0"
                  max={options.chunkSize / 2}
                  className="mff-option-input"
                />
              </label>
            </div>
            
            <div className="mff-checkbox-group">
              <input
                type="checkbox"
                id="clear-index"
                name="clearIndex"
                checked={options.clearIndex}
                onChange={handleOptionChange}
                className="mff-checkbox"
              />
              <label htmlFor="clear-index" className="mff-checkbox-label">
                Clear existing index before uploading
              </label>
            </div>
          </div>

          {error && (
            <div className="mff-upload-error-message">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <div className="mff-upload-actions">
            <button
              type="submit"
              disabled={!file || isUploading}
              className="mff-upload-button"
            >
              {isUploading ? (
                <>
                  <Loader size={16} className="mff-spinner" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={16} />
                  Upload File
                </>
              )}
            </button>
            {file && !isUploading && (
              <button
                type="button"
                className="mff-cancel-button"
                onClick={() => setFile(null)}
              >
                Cancel
              </button>
            )}
          </div>

          <div className="mff-upload-note">
            <p>
              <strong>Notes:</strong>
            </p>
            <ul>
              <li>PDF files will be processed using OCR if needed</li>
              <li>CSV files must include "title" and "contents" (or "text") columns</li>
              <li>JSON files must include "title" and "text" fields</li>
              <li>Maximum file size: 10MB</li>
            </ul>
          </div>
        </form>
      )}
    </div>
  );
}

export default FileUpload;