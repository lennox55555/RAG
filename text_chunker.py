# # text_chunker.py
# from typing import List, Dict, Any
# import re
# import nltk
# from nltk.tokenize import sent_tokenize
# import tiktoken

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# class TextChunker:
#     def __init__(self, 
#                  max_tokens: int = 7500,
#                  chunk_overlap: int = 0,  # Temporarily disable overlap
#                  split_by_paragraph: bool = True):
#         self.max_tokens = max_tokens
#         self.chunk_overlap = chunk_overlap
#         self.split_by_paragraph = split_by_paragraph
#         self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
#     def _normalize_text(self, text: str) -> str:
#         text = re.sub(r'\s+', ' ', text)
#         text = text.strip()
#         return text
    
#     def _count_tokens(self, text: str) -> int:
#         return len(self.tokenizer.encode(text, disallowed_special=()))
    
#     def _split_into_sentences(self, text: str) -> List[str]:
#         return sent_tokenize(self._normalize_text(text))
    
#     def _chunk_text(self, text: str) -> List[str]:
#         """Split text into chunks under max_tokens."""
#         if self._count_tokens(text) <= self.max_tokens:
#             return [text]
        
#         sentences = self._split_into_sentences(text)
#         chunks = []
#         current_chunk = ""
#         current_tokens = 0
        
#         for sentence in sentences:
#             sentence_tokens = self._count_tokens(sentence)
#             if sentence_tokens > self.max_tokens:
#                 # Handle oversized sentences by splitting into words
#                 words = sentence.split()
#                 sub_chunk = ""
#                 sub_tokens = 0
#                 for word in words:
#                     word_tokens = self._count_tokens(word)
#                     if sub_tokens + word_tokens > self.max_tokens:
#                         if sub_chunk:
#                             chunks.append(sub_chunk.strip())
#                         sub_chunk = word
#                         sub_tokens = word_tokens
#                     else:
#                         sub_chunk += " " + word
#                         sub_tokens += word_tokens
#                 if sub_chunk:
#                     chunks.append(sub_chunk.strip())
#                 continue
            
#             if current_tokens + sentence_tokens > self.max_tokens:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = sentence
#                 current_tokens = sentence_tokens
#             else:
#                 current_chunk += " " + sentence
#                 current_tokens += sentence_tokens
        
#         if current_chunk:
#             chunks.append(current_chunk.strip())
        
#         return chunks
    
#     def chunk_by_paragraph(self, text: str) -> List[Dict[str, Any]]:
#         paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\r\s*\r', text)
#         chunks = []
        
#         for paragraph in paragraphs:
#             paragraph = self._normalize_text(paragraph)
#             if not paragraph:
#                 continue
#             paragraph_chunks = self._chunk_text(paragraph)
#             chunks.extend(paragraph_chunks)
        
#         formatted_chunks = []
#         for i, chunk_text in enumerate(chunks):
#             token_count = self._count_tokens(chunk_text)
#             if token_count > self.max_tokens:
#                 print(f"Error: Chunk {i} exceeds max_tokens ({token_count} > {self.max_tokens})")
#             formatted_chunks.append({
#                 "chunk_id": i,
#                 "text": chunk_text,
#                 "token_count": token_count
#             })
        
#         return formatted_chunks
    
#     def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, Any]]:
#         title = document["title"]
#         text = document["text"]
        
#         if not text or len(text) < 5:
#             print(f"Warning: Document '{title}' has insufficient text for chunking.")
#             token_count = self._count_tokens(text)
#             return [{"chunk_id": 0, "text": text, "token_count": token_count, "doc_title": title, "source": title}]
        
#         if self.split_by_paragraph:
#             chunks = self.chunk_by_paragraph(text)
#         else:
#             chunks = self._chunk_text(text)
#             chunks = [{"chunk_id": i, "text": c, "token_count": self._count_tokens(c)} for i, c in enumerate(chunks)]
        
#         for chunk in chunks:
#             chunk["doc_title"] = title
#             chunk["source"] = title
        
#         return chunks
    
#     def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
#         all_chunks = []
        
#         for i, document in enumerate(documents):
#             doc_chunks = self.chunk_document(document)
#             for chunk in doc_chunks:
#                 chunk["doc_index"] = i
#             all_chunks.extend(doc_chunks)
        
#         print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
#         return all_chunks

# if __name__ == "__main__":
#     from data_reader import DataReader
#     from text_extractor import TextExtractor
    
#     reader = DataReader("data/EssaySampleText.json")
#     extractor = TextExtractor(reader)
#     documents = extractor.extract_all_texts()
    
#     chunker = TextChunker(max_tokens=7500, chunk_overlap=0, split_by_paragraph=True)
#     chunks = chunker.chunk_documents(documents)
    
#     if chunks:
#         print(f"Total chunks: {len(chunks)}")
#         for i, chunk in enumerate(chunks[:5]):
#             print(f"Chunk {i}: {chunk['text'][:150]}... (Tokens: {chunk['token_count']})")


# text_chunker.py
# import tiktoken
# from typing import List, Dict, Any
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TextChunker:
#     def __init__(self, max_tokens: int = 7000, overlap: int = 50):
#         """Initialize chunker with maximum tokens and overlap."""
#         self.max_tokens = max_tokens
#         self.overlap = overlap
#         self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

#     def count_tokens(self, text: str) -> int:
#         """Count tokens in the text."""
#         try:
#             return len(self.encoding.encode(text))
#         except Exception as e:
#             logger.error(f"Error counting tokens: {str(e)}")
#             return 0

#     def chunk_text(self, text: str) -> List[Dict[str, Any]]:
#         """Chunk text into pieces with a maximum token limit."""
#         chunks = []
#         tokens = self.encoding.encode(text)
#         total_tokens = len(tokens)

#         if total_tokens <= self.max_tokens:
#             return [{"text": text, "token_count": total_tokens}]

#         start = 0
#         while start < total_tokens:
#             end = min(start + self.max_tokens, total_tokens)
#             chunk_tokens = tokens[start:end]
#             chunk_text = self.encoding.decode(chunk_tokens)
#             chunks.append({
#                 "text": chunk_text,
#                 "token_count": len(chunk_tokens)
#             })
#             start = end - self.overlap
#         return chunks



# text_chunker.py
import tiktoken
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, max_tokens: int = 7000, overlap: int = 50):
        """
        Initialize the TextChunker with maximum tokens per chunk and overlap between chunks.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk (default: 7000).
            overlap (int): Number of tokens to overlap between consecutive chunks (default: 50).
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must be non-negative and less than max_tokens")
        
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text (str): The text to count tokens for.
        
        Returns:
            int: Number of tokens.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk the text into pieces with a maximum token limit and optional overlap.
        
        Args:
            text (str): The text to chunk.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing chunk text and token count.
        """
        chunks = []
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.max_tokens:
            return [{"text": text, "token_count": total_tokens}]

        start = 0
        while start < total_tokens:
            end = min(start + self.max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })
            start = end - self.overlap

        return chunks

# No __main__ block to avoid unintentional execution