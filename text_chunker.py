from typing import List, Dict, Any
import re
import nltk
from nltk.tokenize import sent_tokenize

# nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextChunker:
    
     # class for chunking documents into smaller pieces for embedding and retrieval.
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, split_by_paragraph: bool = True):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by_paragraph = split_by_paragraph
    
    def _normalize_text(self, text: str) -> str:
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def chunk_by_sentence(self, text: str) -> List[Dict[str, Any]]:
    
        # normalize the text
        text = self._normalize_text(text)
        
        # Simple sentence splitting based on common ending punctuation
        # This is less sophisticated than NLTK's sent_tokenize but doesn't require punkt_tab
        simple_sentences = []
        for potential_sentence in re.split(r'(?<=[.!?])\s+', text):
            if potential_sentence:
                simple_sentences.append(potential_sentence)
        
        chunks = []
        current_chunk = ""
        
        for sentence in simple_sentences:
            # If adding this sentence would exceed the chunk size,
            # save the current chunk and start a new one with overlap
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap by keeping some of the previous text
                if self.chunk_overlap > 0:
                    # Keep the last part of the current chunk as overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Find a good place to break in the overlap
                    break_point = max(overlap_text.rfind('.'), overlap_text.rfind('!'), overlap_text.rfind('?'))
                    if break_point > 0:
                        current_chunk = overlap_text[break_point+1:].strip()
                    else:
                        current_chunk = ""
                else:
                    current_chunk = ""
            
            # Add the current sentence
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        # Add the final chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Format the chunks with metadata
        formatted_chunks = []
        for i, chunk_text in enumerate(chunks):
            formatted_chunks.append({
                "chunk_id": i,
                "text": chunk_text,
                "char_length": len(chunk_text)
            })
    
        return formatted_chunks
    
    def chunk_by_paragraph(self, text: str) -> List[Dict[str, Any]]:
    
        # split by paragraph breaks (various newline patterns)
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\r\s*\r', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = self._normalize_text(paragraph)
            
            # Skip empty paragraphs
            if not paragraph:
                continue
                
            # If this paragraph alone exceeds chunk size, use sentence chunking for it
            if len(paragraph) > self.chunk_size:
                paragraph_chunks = self.chunk_by_sentence(paragraph)
                for p_chunk in paragraph_chunks:
                    chunks.append(p_chunk["text"])
                continue
                
            # If adding this paragraph would exceed the chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Format the chunks with metadata
        formatted_chunks = []
        for i, chunk_text in enumerate(chunks):
            formatted_chunks.append({
                "chunk_id": i,
                "text": chunk_text,
                "char_length": len(chunk_text)
            })
        
        return formatted_chunks
    
    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, Any]]:

        title = document["title"]
        text = document["text"]
        
        # Check if document is valid
        if not text or len(text) < 10:
            print(f"Warning: Document '{title}' has insufficient text for chunking.")
            return []
        
        if self.split_by_paragraph:
            chunks = self.chunk_by_paragraph(text)
        else:
            chunks = self.chunk_by_sentence(text)
        
        for chunk in chunks:
            chunk["doc_title"] = title
            chunk["source"] = title
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:

        all_chunks = []
        
        for i, document in enumerate(documents):
            doc_chunks = self.chunk_document(document)
            
            # Add document index to each chunk
            for chunk in doc_chunks:
                chunk["doc_index"] = i
            
            all_chunks.extend(doc_chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


if __name__ == "__main__":
    from data_reader import DataReader
    from text_extractor import TextExtractor
    
    reader = DataReader("data/EssaySampleText.json")
    extractor = TextExtractor(reader)
    documents = extractor.extract_all_texts()
    
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200, split_by_paragraph=True)
    chunks = chunker.chunk_documents(documents)
    
    if chunks:
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk: {chunks[0]['text'][:150]}...")