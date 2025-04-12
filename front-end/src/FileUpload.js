import React, { useState, useEffect, useRef } from 'react';
import { Upload, CheckCircle, AlertCircle, FileText, Database, Loader, X, Package, File } from 'lucide-react';
import { uploadFile, getUploadStatus } from './api';

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
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  const [processingMessage, setProcessingMessage] = useState('');
  
  // For polling upload status
  const pollingInterval = useRef(null);

  // Effect for polling task status
  useEffect(() => {
    // Clean up polling on unmount
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, []);
  
  // Start polling when taskId is set
  useEffect(() => {
    if (taskId) {
      startPollingStatus(taskId);
    }
    
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
        pollingInterval.current = null;
      }
    };
  }, [taskId]);
  
  // Poll for upload status
  const startPollingStatus = (id) => {
    // Stop any existing polling
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
    }
    
    // Check immediately then start polling
    checkUploadStatus(id);
    
    pollingInterval.current = setInterval(() => {
      checkUploadStatus(id);
    }, 2000); // Check every 2 seconds
  };
  
  const checkUploadStatus = async (id) => {
    try {
      const status = await getUploadStatus(id);
      console.log("Status update:", status); // Debug log
      
      // Update UI with latest status
      setProgress(status.progress || 0);
      setProcessingStatus(status.status || 'processing');
      setProcessingMessage(status.message || 'Processing...');
      
      // If complete, stop polling and set result
      if (status.is_complete) {
        console.log("Processing complete:", status); // Debug log
        
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current);
          pollingInterval.current = null;
        }
        
        setIsUploading(false);
        setTaskId(null);
        
        // Create result object for display
        const result = {
          status: status.status || 'complete',
          message: status.message || 'Processing completed',
          filename: file ? file.name : 'document',
          documents_processed: status.documents_processed || 0,
          chunks_created: status.chunks_created || 0,
          chunks_indexed: status.chunks_indexed || 0
        };
        
        // Special handling for error states
        if (status.status === 'error') {
          result.status = 'error';
          console.error("Upload processing failed:", status.message);
        }
        
        setUploadResult(result);
      }
    } catch (err) {
      console.error('Error checking upload status:', err);
      
      // After several retries, assume server error and stop polling
      const failedChecks = (pollingInterval.retryCount || 0) + 1;
      pollingInterval.retryCount = failedChecks;
      
      if (failedChecks > 5) {
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current);
          pollingInterval.current = null;
        }
        
        setIsUploading(false);
        setError(`Lost connection to server while processing: ${err.message}`);
      }
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setUploadResult(null);
      setTaskId(null);
      setProgress(0);
      setProcessingStatus('');
      setProcessingMessage('');
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
    if (!['pdf', 'csv', 'json', 'zip'].includes(fileExtension)) {
      setError('Only PDF, CSV, JSON, and ZIP files are supported');
      return;
    }

    setIsUploading(true);
    setError(null);
    setProgress(0);
    setProcessingStatus('uploading');
    setProcessingMessage('Uploading file...');
    
    try {
      const result = await uploadFile(file, options);
      
      // If we get a task ID, start polling for status
      if (result.task_id) {
        setTaskId(result.task_id);
        // Don't set upload result yet, we'll get it from polling
      } else {
        // We got an immediate result (unlikely with new implementation)
        setUploadResult(result);
        setIsUploading(false);
        setFile(null);
        // Reset file input
        e.target.reset();
      }
    } catch (err) {
      setError(err.message || 'An error occurred while uploading the file');
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setUploadResult(null);
    setError(null);
    setTaskId(null);
    setProgress(0);
    setProcessingStatus('');
    setProcessingMessage('');
    
    // Clear any polling interval
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  return (
    <div className="mff-file-upload">
      <h3 className="mff-section-title">Upload Documents</h3>
      <p className="mff-section-description">
        Upload PDF, CSV, or JSON files to add documents to the knowledge base
      </p>

      {uploadResult ? (
        <div className={`mff-upload-result ${uploadResult.status === 'error' ? 'mff-upload-error' : 'mff-upload-success'}`}>
          {uploadResult.status === 'error' ? (
            <AlertCircle className="mff-result-icon mff-error-icon" size={40} />
          ) : (
            <CheckCircle className="mff-result-icon mff-success-icon" size={40} />
          )}

          <h4 className="mff-result-title">
            {uploadResult.status === 'error' ? 'Upload Failed' : 'Upload Successful'}
          </h4>
          
          <p className="mff-result-message">{uploadResult.message}</p>
          
          {uploadResult.chunks_created > 0 && (
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
                accept=".pdf,.csv,.json,.zip"
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
                  {processingStatus === 'uploading' ? 'Uploading...' : 'Processing...'}
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

          {/* Progress bar for uploads */}
          {isUploading && (
            <div className="mff-upload-progress">
              <div className="mff-progress-bar-container">
                <div 
                  className="mff-progress-bar" 
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <div className="mff-progress-info">
                <div className="mff-progress-detail">
                  <span className="mff-progress-percentage">{Math.round(progress)}%</span>
                  <span className="mff-progress-stage">{processingStatus}</span>
                </div>
                <span className="mff-progress-status">{processingMessage}</span>
              </div>
            </div>
          )}
          
          <div className="mff-upload-note">
            <p>
              <strong>Notes:</strong>
            </p>
            <ul>
              <li>PDF files will be processed using OCR if needed</li>
              <li>CSV files must include "title" and "contents" (or "text") columns</li>
              <li>JSON files must include "title" and "text" fields</li>
              <li>ZIP files can contain multiple PDFs that will all be processed</li>
              <li>No file size limit - large files will be processed in the background</li>
            </ul>
          </div>
        </form>
      )}
    </div>
  );
}

export default FileUpload;