import os
import argparse
import logging
import json
from dotenv import load_dotenv
from document_processor import DataReader, TextExtractor, TextChunker, PDFProcessor
from embedding_service import EmbeddingService
from rag_pipeline import create_rag_pipeline_from_env, RAGPipeline

# setup logging
from logger_config import setup_logger
logger = setup_logger("main")

# load env vars
load_dotenv()

def process_pdfs(input_folder: str, output_folder: str):
    # process pdfs to json files
    logger.info(f"Processing PDFs from {input_folder} to {output_folder}")
    PDFProcessor.process_pdfs_to_json(input_folder, output_folder)

def process_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    # process docs into chunks
    logger.info(f"Processing documents from {file_path}")
    
    # read and extract
    reader = DataReader(file_path)
    extractor = TextExtractor(reader)
    documents = extractor.extract_all_texts()
    
    logger.info(f"Extracted {len(documents)} documents")
    
    # create chunks
    chunker = TextChunker(max_tokens=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks

def index_documents(file_paths: list, chunk_size: int = 1000, chunk_overlap: int = 100, 
                   clear_index: bool = False, save_embeddings: bool = False):
    # index documents in vector db
    logger.info(f"Indexing {len(file_paths)} document sources")
    
    # init embedding service
    try:
        embedding_service = EmbeddingService()
        
        # clear index if requested
        if clear_index:
            logger.info("Clearing Pinecone index")
            embedding_service.clear_pinecone_index()
        
        # process each file
        all_chunks = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            logger.info(f"Processing {file_path}")
            chunks = process_documents(file_path, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks generated from input files")
            return
            
        # create and upload embeddings
        logger.info(f"Creating embeddings and uploading {len(all_chunks)} chunks")
        
        save_path = "out/embeddings.json" if save_embeddings else None
        embedding_service.process_and_upload(all_chunks, save_path=save_path)
        
        logger.info("Indexing complete")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

def test_query(query: str, with_citations: bool = True):
    # test query against rag system
    logger.info(f"Testing query: {query}")
    
    try:
        # create pipeline
        pipeline = create_rag_pipeline_from_env()
        
        # process query
        result = pipeline.process_query(query, with_citations=with_citations)
        
        # print result
        logger.info(f"Query: {query}")
        logger.info(f"Response: {result['response']}")
        logger.info(f"Retrieved {result['num_docs_retrieved']} documents")
        logger.info(f"Confidence: {result['confidence']['level']} ({result['confidence']['score']:.2f})")
        
        # print retrieved docs
        if result['retrieved_docs']:
            logger.info("\nRetrieved Documents:")
            for i, doc in enumerate(result['retrieved_docs'], 1):
                logger.info(f"\n[{i}] {doc['doc_title']} - {doc['similarity_category']} ({int(doc['similarity']*100)}% match)")
                preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
                logger.info(f"Preview: {preview}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing query: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Mary Ferrell Foundation RAG Application")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # process pdfs command
    pdf_parser = subparsers.add_parser("process-pdfs", help="Process PDF files into JSON")
    pdf_parser.add_argument("--input", required=True, help="Folder containing PDF files")
    pdf_parser.add_argument("--output", required=True, help="Output folder for JSON files")
    
    # index documents command
    index_parser = subparsers.add_parser("index", help="Index documents in vector database")
    index_parser.add_argument("--files", nargs="+", required=True, help="JSON files to process")
    index_parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size in tokens")
    index_parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in tokens")
    index_parser.add_argument("--clear-index", action="store_true", help="Clear index before indexing")
    index_parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to disk")
    
    # test query command
    query_parser = subparsers.add_parser("query", help="Test the RAG pipeline with a query")
    query_parser.add_argument("--query", required=True, help="Query string to test")
    query_parser.add_argument("--no-citations", action="store_true", help="Don't include citations in response")
    query_parser.add_argument("--output", help="Optional JSON file to save results to")
    
    # parse args
    args = parser.parse_args()
    
    # execute command
    if args.command == "process-pdfs":
        process_pdfs(args.input, args.output)
    elif args.command == "index":
        index_documents(args.files, args.chunk_size, args.chunk_overlap, args.clear_index, args.save_embeddings)
    elif args.command == "query":
        result = test_query(args.query, not args.no_citations)
        
        # save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved results to {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()