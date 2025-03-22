# from typing import List, Dict, Any
# from data_reader import DataReader

# class TextExtractor:
    
#     # extracting text from document JSON objects in data folder. only a few partial documents are uplaoded to live site
    
#     def __init__(self, reader: DataReader):
    
#         self.reader = reader
    
#     def extract_document_text(self, document: Dict[str, Any]) -> Dict[str, str]:

#         title = document.get("Title", "")
#         text = document.get("Text", "")
        
#         if text == "" and title.startswith("Error -"):
#             text = "No content available for this document."
        
#         return {
#             "title": title,
#             "text": text
#         }
    
#     def extract_all_texts(self) -> List[Dict[str, str]]:
#         documents = self.reader.get_all_documents()
#         extracted_texts = []
        
#         for doc in documents:
#             extracted = self.extract_document_text(doc)
#             if extracted["title"] and extracted["text"]: 
#                 extracted_texts.append(extracted)
        
#         print(f"Extracted text from {len(extracted_texts)} documents")
#         return extracted_texts


# if __name__ == "__main__":
#     reader = DataReader("data/EssaySampleText.json")
#     extractor = TextExtractor(reader)
#     texts = extractor.extract_all_texts()
    
#     if texts:
#         first_doc = texts[0]
#         print(f"Title: {first_doc['title']}")
#         print(f"Text sample: {first_doc['text'][:150]}...")



# text_extractor.py
from typing import List, Dict, Any
from data_reader import DataReader

class TextExtractor:
    """
    Class for extracting text from document JSON objects.
    """
    def __init__(self, reader: DataReader):
        """
        Initialize the TextExtractor with a DataReader.
        
        Args:
            reader: DataReader instance to access the data
        """
        self.reader = reader
    
    def extract_document_text(self, document: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract the title and text from a document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Dictionary with title and text
        """
        title = document.get("Title", "")
        text = document.get("Text", "")
        
        # Handle empty documents or error pages
        if text == "" and title.startswith("Error -"):
            text = "No content available for this document."
        
        return {
            "title": title,
            "text": text
        }
    
    def extract_all_texts(self) -> List[Dict[str, str]]:
        """
        Extract text from all documents.
        
        Returns:
            List of dictionaries with title and text for each document
        """
        documents = self.reader.get_all_documents()
        extracted_texts = []
        
        for doc in documents:
            extracted = self.extract_document_text(doc)
            if extracted["title"]:  # Only include if there's a title
                extracted_texts.append(extracted)
        
        print(f"Extracted text from {len(extracted_texts)} documents")
        return extracted_texts

if __name__ == "__main__":
    reader = DataReader("data/EssaySampleText.json")
    extractor = TextExtractor(reader)
    texts = extractor.extract_all_texts()
    
    if texts:
        first_doc = texts[0]
        print(f"Title: {first_doc['title']}")
        print(f"Text sample: {first_doc['text'][:150]}...")