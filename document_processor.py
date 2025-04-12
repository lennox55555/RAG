import os
import json
import csv
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from wordsegment import load, segment

# setup logging
from logger_config import setup_logger
logger = setup_logger("document_processor")

# load word segmentation dictionary
load()

class DataReader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def read_data(self) -> List[Dict[str, Any]]:
        # read json data from file
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    logger.error(f"Error: {self.data_path} does not contain a JSON list")
                    return []
                logger.info(f"Loaded {len(data)} documents from {self.data_path}")
                return data
        except FileNotFoundError:
            logger.error(f"Error: File {self.data_path} not found")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error: File {self.data_path} is not valid JSON")
            logger.error(f"JSON error details: {e}")
            try:
                with open(self.data_path, 'r', encoding='utf-8') as file:
                    first_lines = [next(file) for _ in range(10)]
                    logger.debug(f"First few lines: {''.join(first_lines)}")
            except Exception as file_error:
                logger.error(f"Error reading file for debug: {file_error}")
            return []
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            return []   
    
    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        # get document by index
        data = self.read_data()
        if 0 <= doc_id < len(data):
            return data[doc_id]
        return {}
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        # get all documents
        return self.read_data()

class PDFProcessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        # separate run-together words with spaces
        lines = text.split("\n")
        processed_lines = []
        
        for line in lines:
            # skip empty lines
            if not line.strip():
                processed_lines.append("")
                continue
            
            # segment words
            words = segment(line)
            processed_line = " ".join(words)
            processed_lines.append(processed_line)
        
        return "\n".join(processed_lines)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        extract text from pdf with page tracking and progress updates
        
        args:
            pdf_path: path to pdf file
            progress_callback: callback for progress updates
                              func(stage, current, total)
        """
        pages_text = {}
        full_text = ""
        total_pages = 0
        
        try:
            # try direct text extraction first
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # report initial progress
                if progress_callback:
                    progress_callback('extract', 0, total_pages)
                
                # process each page
                for page_idx, page in enumerate(reader.pages):
                    page_num = page_idx + 1  # use 1-based page numbers
                    
                    # update progress
                    if progress_callback:
                        progress_callback('extract', page_num, total_pages)
                    
                    # get text from page
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        # process text for this page
                        processed_page_text = PDFProcessor.preprocess_text(page_text.strip())
                        pages_text[page_num] = processed_page_text
                        full_text += processed_page_text + "\n"
            
            # fallback to ocr if no text found
            if not full_text.strip():
                logger.info(f"No text extracted directly from {pdf_path}, using OCR...")
                images = convert_from_path(pdf_path)
                total_pages = len(images)
                
                # report initial ocr progress
                if progress_callback:
                    progress_callback('ocr', 0, total_pages)
                
                # process each page with ocr
                for page_idx, image in enumerate(images):
                    page_num = page_idx + 1  # use 1-based page numbers
                    
                    # update ocr progress
                    if progress_callback:
                        progress_callback('ocr', page_num, total_pages)
                    
                    # perform ocr on image
                    page_text = pytesseract.image_to_string(image)
                    
                    if page_text and page_text.strip():
                        # process ocr text
                        processed_page_text = PDFProcessor.preprocess_text(page_text.strip())
                        pages_text[page_num] = processed_page_text
                        full_text += processed_page_text + "\n"
            
            # return extracted text with page information
            return {
                "full_text": full_text.strip(),
                "pages_text": pages_text,
                "total_pages": total_pages
            }
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return {
                "full_text": "",
                "pages_text": {},
                "total_pages": 0
            }

    @staticmethod
    def process_pdfs_to_json(pdf_folder: str, output_folder: str):
        # process all pdfs to json files
        pdf_folder_path = Path(pdf_folder)
        output_folder_path = Path(output_folder)
        
        if not pdf_folder_path.exists() or not pdf_folder_path.is_dir():
            raise ValueError(f"PDF folder {pdf_folder} does not exist or is not a directory")
        
        # create output folder
        output_folder_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        # process each pdf
        for pdf_file in pdf_folder_path.glob("*.pdf"):
            pdf_path = str(pdf_file)
            filename = pdf_file.stem
            logger.info(f"Processing PDF: {pdf_path}")

            # extract text
            content = PDFProcessor.extract_text_from_pdf(pdf_path)
            if not content:
                logger.warning(f"No content extracted from {pdf_path}, skipping...")
                continue

            # create doc entry
            doc = {
                "title": filename,
                "text": content,
                "key": filename,
                "source": filename
            }

            # save to json file
            output_json_path = output_folder_path / f"{filename}.json"
            with open(output_json_path, 'w') as f:
                json.dump(doc, f, indent=2)
            
            processed_count += 1
            logger.info(f"Saved document to {output_json_path}")

        logger.info(f"Processed {processed_count} PDFs and saved to {output_folder}")

class TextExtractor:
    def __init__(self, reader: DataReader):
        self.reader = reader
    
    def extract_document_text(self, document: Dict[str, Any]) -> Dict[str, str]:
        # extract title and text
        title = document.get("Title", document.get("title", ""))
        text = document.get("Text", document.get("text", ""))
        source = document.get("Source", document.get("source", ""))
        key = document.get("Key", document.get("key", ""))
        
        # handle empty docs
        if text == "" and title.startswith("Error -"):
            text = "No content available for this document."
        
        return {
            "title": title,
            "text": text,
            "source": source,
            "key": key
        }
    
    def extract_all_texts(self) -> List[Dict[str, str]]:
        # extract from all docs
        documents = self.reader.get_all_documents()
        extracted_texts = []
        
        for doc in documents:
            extracted = self.extract_document_text(doc)
            if extracted["title"]:  
                extracted_texts.append(extracted)
        
        logger.info(f"Extracted text from {len(extracted_texts)} documents")
        return extracted_texts

class TextChunker:
    def __init__(self, max_tokens=7000, chunk_overlap=200):
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
    
    def estimate_token_count(self, text: str) -> int:
        # estimate tokens using word count
        words = len(text.split())
        return int(words * 1.3)  # safety margin
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        # split text into chunks
        tokens = self.estimate_token_count(text)
        
        # return single chunk if small enough
        if tokens <= self.max_tokens:
            return [{
                "text": text,
                "token_count": tokens
            }]
        
        chunks = []
        # split by sentences
        sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")
        
        current_chunk = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_token_count(sentence)
            
            # if sentence too long, split by words
            if sentence_tokens > self.max_tokens:
                words = sentence.split()
                
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
            
            # check if adding sentence exceeds limit
            if current_chunk_tokens + sentence_tokens > self.max_tokens:
                # save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_chunk_tokens
                })
                
                # start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 0:
                    # calculate overlap sentences
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
        
        # add last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "token_count": current_chunk_tokens
            })
        
        return chunks
    
    def find_page_for_chunk(self, chunk_text: str, pages_dict: Dict[int, str]) -> int:
        """Find the page number that contains the majority of this chunk's text"""
        if not pages_dict:
            return 0
            
        # If only one page, it must be from that page
        if len(pages_dict) == 1:
            return list(pages_dict.keys())[0]
            
        # Score each page by how much of the chunk text appears in it
        page_scores = {}
        
        # Simplify chunk text for matching (remove whitespace variations)
        simplified_chunk = ' '.join(chunk_text.split())
        
        for page_num, page_text in pages_dict.items():
            # Simplify page text
            simplified_page = ' '.join(page_text.split())
            
            # Check if chunk appears in page
            if simplified_chunk in simplified_page:
                # Exact match, definitely from this page
                return page_num
                
            # Calculate how many characters of chunk appear in this page
            # Use a simple word-based overlap metric
            chunk_words = set(simplified_chunk.lower().split())
            page_words = set(simplified_page.lower().split())
            common_words = chunk_words.intersection(page_words)
            
            if common_words:
                page_scores[page_num] = len(common_words) / len(chunk_words)
        
        # Return page with highest score, or first page if none match
        if page_scores:
            return max(page_scores.items(), key=lambda x: x[1])[0]
        
        # If we can't determine, use first page
        return min(pages_dict.keys())

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # process docs into chunks
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            title = doc.get("title", "")
            text = doc.get("text", "")
            source = doc.get("source", "")
            key = doc.get("key", f"doc_{doc_idx}")
            pages = doc.get("pages", {})  # Dictionary mapping page numbers to text
            total_pages = doc.get("total_pages", 0)
            
            # skip empty docs
            if not text:
                continue
                
            # combine title and text
            full_text = f"{title}: {text}" if title else text
            
            # chunk text
            chunked = self.chunk_text(full_text)
            
            # add metadata
            for chunk_idx, chunk in enumerate(chunked):
                chunk_text = chunk["text"]
                
                # Determine which page this chunk comes from (if PDF)
                page_num = 0
                if pages:
                    page_num = self.find_page_for_chunk(chunk_text, pages)
                
                chunk["doc_index"] = doc_idx
                chunk["chunk_id"] = chunk_idx
                chunk["doc_key"] = key
                chunk["doc_title"] = title
                chunk["source"] = source
                chunk["page_num"] = page_num
                chunk["total_pages"] = total_pages
                
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# test
if __name__ == "__main__":
    # test pdf processing
    pdf_folder = "data/report_pdfs"
    output_folder = "data/pdf_extracted"
    
    try:
        PDFProcessor.process_pdfs_to_json(pdf_folder, output_folder)
        
        # test document extraction
        reader = DataReader("data/EssaySampleText.json")
        extractor = TextExtractor(reader)
        texts = extractor.extract_all_texts()
        
        # test chunking
        chunker = TextChunker(max_tokens=1000, chunk_overlap=100)
        chunks = chunker.chunk_documents(texts)
        
        logger.info(f"Generated {len(chunks)} chunks from {len(texts)} documents")
        
        # print sample
        if chunks:
            sample = chunks[0]
            logger.info(f"Sample chunk - Title: {sample['doc_title']}, Tokens: {sample['token_count']}")
            logger.info(f"Sample text: {sample['text'][:150]}...")
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")