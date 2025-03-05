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
                return data
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []
        except Exception as e:
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
    