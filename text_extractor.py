from typing import List, Dict, Any
from data_reader import DataReader

class TextExtractor:
    """
    Extracting text from document JSON objects.
    """
    def __init__(self, reader: DataReader):
    
        self.reader = reader
    
    def extract_document_text(self, document: Dict[str, Any]) -> Dict[str, str]:

        title = document.get("Title", "")
        text = document.get("Text", "")
        
        if text == "" and title.startswith("Error -"):
            text = "No content available for this document."
        
        return {
            "title": title,
            "text": text
        }
    
    def extract_all_texts(self) -> List[Dict[str, str]]:
        documents = self.reader.get_all_documents()
        extracted_texts = []
        
        for doc in documents:
            extracted = self.extract_document_text(doc)
            if extracted["title"] and extracted["text"]: 
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