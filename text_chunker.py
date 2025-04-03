# text_chunker.py
from typing import List, Dict, Any

class TextChunker:
    """
    Class to chunk text documents into smaller pieces for embedding.
    """
    def __init__(self, max_tokens=7000, chunk_overlap=200):
        """
        Initialize the TextChunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text using word count approximation.
        
        Args:
            text: The text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple approximation: 1 token ≈ 0.75 words
        words = len(text.split())
        return int(words * 1.3)  # Add a safety margin
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on token limits.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of chunk dictionaries with text and token count
        """
        tokens = self.estimate_token_count(text)
        
        # If text is short enough, return as a single chunk
        if tokens <= self.max_tokens:
            return [{
                "text": text,
                "token_count": tokens
            }]
        
        chunks = []
        # Split by sentences first (crude approximation)
        sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")
        
        current_chunk = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_token_count(sentence)
            
            # If a single sentence is already too long, split it by words
            if sentence_tokens > self.max_tokens:
                words = sentence.split()
                word_chunks = []
                
                for i in range(0, len(words), self.max_tokens // 2):
                    word_chunk = " ".join(words[i:i + self.max_tokens])
                    word_chunk_tokens = self.estimate_token_count(word_chunk)
                    
                    chunks.append({
                        "text": word_chunk,
                        "token_count": word_chunk_tokens
                    })
                
                current_chunk = []
                current_chunk_tokens = 0
                continue
            
            # Check if adding this sentence would exceed the limit
            if current_chunk_tokens + sentence_tokens > self.max_tokens:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_chunk_tokens
                })
                
                # Start a new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    
                    for s in reversed(current_chunk):
                        s_tokens = self.estimate_token_count(s)
                        if overlap_tokens + s_tokens > self.chunk_overlap:
                            break
                        
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    
                    current_chunk = overlap_sentences
                    current_chunk_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_chunk_tokens = 0
            
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "token_count": current_chunk_tokens
            })
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of documents and chunk their text fields.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            title = doc.get("title", "")
            text = doc.get("text", "")
            source = doc.get("source", "")
            key = doc.get("key", f"doc_{doc_idx}")
            
            # Skip empty documents
            if not text:
                continue
                
            # Combine title and text
            full_text = f"{title}: {text}" if title else text
            
            # Chunk the text
            chunked = self.chunk_text(full_text)
            
            # Add metadata to chunks
            for chunk_idx, chunk in enumerate(chunked):
                chunk["doc_index"] = doc_idx
                chunk["chunk_id"] = chunk_idx
                chunk["doc_key"] = key
                chunk["doc_title"] = title
                chunk["source"] = source
                
                all_chunks.append(chunk)
        
        return all_chunks


if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(max_tokens=1000, chunk_overlap=100)
    
    test_text = """This is a sample text for testing the chunker. It contains multiple sentences that should be split into different chunks based on token count. This is a long paragraph with many sentences. Each sentence adds more tokens. We want to make sure the chunker works correctly. This is another sentence to add more tokens. And another one. And one more to make it longer."""
    
    chunks = chunker.chunk_text(test_text)
    
    print(f"Original text has approximately {chunker.estimate_token_count(test_text)} tokens")
    print(f"Split into {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk['token_count']} tokens")
        print(chunk['text'])
        print("-" * 50)