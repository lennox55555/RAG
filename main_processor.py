import os
import argparse
from dotenv import load_dotenv
from data_reader import DataReader
from text_extractor import TextExtractor
from text_chunker import TextChunker
from embedding_creator import EmbeddingCreator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process documents, create embeddings, and upload to Pinecone')
    
    parser.add_argument('--data_path', type=str, default='data/EssaySampleText.json',
                        help='Path to the JSON data file')
    parser.add_argument('--embeddings_path', type=str, default='data/embeddings.json',
                        help='Path to save the embeddings')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Maximum size of each chunk in characters')
    parser.add_argument('--chunk_overlap', type=int, default=200,
                        help='Overlap between consecutive chunks in characters')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of batches for embedding creation and uploads')
    parser.add_argument('--pinecone_namespace', type=str, default='documents',
                        help='Namespace in the Pinecone index')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Whether to save embeddings to a file')
    
    return parser.parse_args()

def main():
    """Main function to run the entire pipeline."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get command line arguments
    args = parse_args()
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("Warning: PINECONE_API_KEY environment variable not set")
        pinecone_api_key = "pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp"
    
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    if not pinecone_index_name:
        pinecone_index_name = "mff"
        print(f"Using default Pinecone index name: {pinecone_index_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.embeddings_path), exist_ok=True)
    
    # Step 1: Read data
    print("\n=== Step 1: Reading Data ===")
    reader = DataReader(args.data_path)
    
    # Step 2: Extract text
    print("\n=== Step 2: Extracting Text ===")
    extractor = TextExtractor(reader)
    documents = extractor.extract_all_texts()
    print(f"Extracted {len(documents)} documents")
    
    # Step 3: Chunk documents
    print("\n=== Step 3: Chunking Documents ===")
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        split_by_paragraph=True
    )
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Step 4: Create embeddings and upload to Pinecone
    print("\n=== Step 4: Creating Embeddings and Uploading to Pinecone ===")
    embedding_creator = EmbeddingCreator(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=args.pinecone_namespace,
        openai_api_key=openai_api_key,
        batch_size=args.batch_size
    )
    
    # Save embeddings if requested
    embeddings_path = args.embeddings_path if args.save_embeddings else None
    embedding_creator.process_and_upload(chunks, save_path=embeddings_path)
    
    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()