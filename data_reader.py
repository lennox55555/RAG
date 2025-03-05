import json
import os
from typing import List, Dict, Any

class DataReader:

    # class for reading and accessing the raw data sources.
 
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def read_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"Successfully loaded {len(data)} documents from {self.data_path}")
                return data
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: File {self.data_path} is not valid JSON")
            return []
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        data = self.read_data()
        if 0 <= doc_id < len(data):
            return data[doc_id]
        return {}
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.read_data()

if __name__ == "__main__":
    reader = DataReader("data/EssaySampleText.json")
    docs = reader.read_data()
    