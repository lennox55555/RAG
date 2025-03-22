# import json
# import os
# from typing import List, Dict, Any

# class DataReader:

#     # class for reading and accessing the raw data sources.
 
#     def __init__(self, data_path: str):
#         self.data_path = data_path
        
#     def read_data(self) -> List[Dict[str, Any]]:
#         try:
#             with open(self.data_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#                 return data
#         except FileNotFoundError:
#             return []
#         except json.JSONDecodeError:
#             return []
#         except Exception as e:
#             return []
    
#     def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
#         data = self.read_data()
#         if 0 <= doc_id < len(data):
#             return data[doc_id]
#         return {}
    
#     def get_all_documents(self) -> List[Dict[str, Any]]:
#         return self.read_data()

# if __name__ == "__main__":
#     reader = DataReader("data/EssaySampleText.json")
#     docs = reader.read_data()
    
# data_reader.py
import json
import os
from typing import List, Dict, Any

class DataReader:
    """
    Class for reading and accessing the raw data sources.
    """
    def __init__(self, data_path: str):
        """
        Initialize the DataReader with the path to the data file.
        
        Args:
            data_path: Path to the JSON data file
        """
        self.data_path = data_path
        
    def read_data(self) -> List[Dict[str, Any]]:
        """
        Read the JSON data file and return its contents.
        
        Returns:
            List of document dictionaries
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    print(f"Error: {self.data_path} does not contain a JSON list")
                    return []
                print(f"Successfully loaded {len(data)} documents from {self.data_path}")
                return data
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: File {self.data_path} is not valid JSON")
            print(f"JSON error details: {e}")
            try:
                with open(self.data_path, 'r', encoding='utf-8') as file:
                    first_lines = [next(file) for _ in range(10)]
                    print(f"First few lines of the file:\n{''.join(first_lines)}")
            except Exception as file_error:
                print(f"Error reading file for debug: {file_error}")
            return []
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            return []   
    
    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        """
        Get a document by its index.
        
        Args:
            doc_id: Index of the document
            
        Returns:
            Document dictionary or empty dict if not found
        """
        data = self.read_data()
        if 0 <= doc_id < len(data):
            return data[doc_id]
        return {}
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents.
        
        Returns:
            List of all document dictionaries
        """
        return self.read_data()

if __name__ == "__main__":
    reader = DataReader("data/EssaySampleText.json")
    docs = reader.read_data()
    if docs:
        print(f"Number of docs: {len(docs)}")
        print(f"First doc: {docs[0]}")
    else:
        print("No documents loaded")