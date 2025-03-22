import csv
import traceback
from typing import List, Dict, Any

def parse_essays_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the CSV file with robust error handling
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries with essay information
    """
    essays = []
    
    try:
        # Read the CSV file manually to handle irregularities
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip the header line
            header = next(f).strip().split(',')
            
            line_num = 1  # Header was line 1
            for line in f:
                line_num += 1
                
                try:
                    # Use the csv module to properly handle quotes and commas
                    row = next(csv.reader([line]))
                    
                    # Make sure we have enough values for type, key, title, url
                    if len(row) >= 4:
                        essay = {
                            'type': row[0],
                            'key': row[1],
                            'title': row[2],
                            'url': row[3]
                        }
                        
                        # Only add essays with valid URLs
                        if essay['type'] == 'essay' and essay['url']:
                            essays.append(essay)
                    else:
                        print(f"Warning: Line {line_num} has fewer than 4 fields: {line.strip()}")
                        
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    print(f"Line content: {line.strip()}")
                    continue
                    
        print(f"Successfully parsed {len(essays)} essays from CSV")
        return essays
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        print(traceback.format_exc())
        return []

if __name__ == "__main__":
    # Example usage
    essays = parse_essays_csv("data/essays-mff-duke.csv")
    print(f"Found {len(essays)} essays")
    
    if essays:
        for i, essay in enumerate(essays[:5]):
            print(f"Essay {i+1}: {essay['title']} - {essay['url']}")