import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import os
import time
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import random
from dotenv import load_dotenv
from text_chunker import TextChunker
from embedding_creator import EmbeddingCreator
import csv
import traceback

# import the CSV parser
from csv_parser import parse_essays_csv

# disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# configure headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
}

def scrape_url(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.extract()
        
        # Get text and remove extra whitespace
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        print(f"Successfully scraped {url} ({len(text)} characters)")
        return text
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

def scrape_pdf_url(url: str) -> str:
    print(f"PDF URL detected: {url}")
    
    try:
        try:
            from pdf_handler import scrape_pdf_url as pdf_handler_scrape
            return pdf_handler_scrape(url, HEADERS)
        except ImportError:
            pass
        
        response = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        response.raise_for_status()
        
        pdf_size = len(response.content)
        return f"[This is PDF content from {url}, size: {pdf_size/1024:.1f} KB. Please use a PDF parser to extract the actual content.]"
    except Exception as e:
        print(f"Error accessing PDF at {url}: {str(e)}")
        return f"[Unable to access PDF at {url}: {str(e)}]"

def read_csv(csv_path: str) -> pd.DataFrame:
    
    essays = parse_essays_csv(csv_path)
    if essays:
        return pd.DataFrame(essays)
    return pd.DataFrame()

def read_json(json_path: str) -> List[Dict[str, str]]:
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file {json_path} not found, creating a new one.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {str(e)}")
        return []

def update_json(json_path: str, documents: List[Dict[str, str]]):
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"Error updating JSON file: {str(e)}")

def get_existing_titles(documents: List[Dict[str, str]]) -> set:
    
    return {doc["Title"] for doc in documents}

def main():
    parser = argparse.ArgumentParser(description='Scrape essays from URLs in a CSV and update the database.')
    parser.add_argument('--csv_path', type=str, default='essays.csv', help='Path to the CSV file with essay URLs')
    parser.add_argument('--json_path', type=str, default='data/EssaySampleText.json', help='Path to the JSON database file')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between scraping URLs (seconds)')
    parser.add_argument('--upload', action='store_true', help='Upload to Pinecone after scraping')
    parser.add_argument('--force', action='store_true', help='Force re-scrape of existing documents')
    args = parser.parse_args()
    
    # ensure output directory exists
    os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
    
    # read existing documents
    documents = read_json(args.json_path)
    existing_titles = get_existing_titles(documents)
    print(f"Found {len(documents)} existing documents in {args.json_path}")
    
    # read CSV file with essay URLs
    df = read_csv(args.csv_path)
    if df.empty:
        print("No essays found in CSV file.")
        return
    
    print(f"Found {len(df)} essays in CSV file.")
    
    # scrape each URL and add to documents
    new_documents = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scraping essays"):
        title = row.get('title')
        url = row.get('url')
        
        if not title or not url:
            continue
        
        if title in existing_titles and not args.force:
            print(f"Skipping {title} (already exists)")
            continue
        
        # determine if this is a PDF or HTML URL
        if url.lower().endswith('.pdf'):
            text = scrape_pdf_url(url)
        else:
            text = scrape_url(url)
        
        if text:
            new_documents.append({
                "Title": title,
                "Text": text,
                "Source": url
            })
            
            # add random delay to avoid overwhelming the server
            delay = args.delay + random.uniform(0, 1.0)
            time.sleep(delay)
    
    # add new documents to the existing ones
    if new_documents:
        documents.extend(new_documents)
        update_json(args.json_path, documents)
        print(f"Added {len(new_documents)} new documents to {args.json_path}")
    else:
        print("No new documents to add.")
    
    # if upload flag is set, process and upload to Pinecone
    if args.upload and new_documents:
        print("\n=== Processing and uploading new documents to Pinecone ===")
        
        # load environment variables
        load_dotenv()
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
            return
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("Warning: PINECONE_API_KEY environment variable not set")
            pinecone_api_key = "pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp"
        
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "mff")
        
        # process new documents only
        processed_docs = [{"title": doc["Title"], "text": doc["Text"]} for doc in new_documents]
        
        # chunk documents
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200, split_by_paragraph=True)
        chunks = chunker.chunk_documents(processed_docs)
        print(f"Created {len(chunks)} chunks from {len(processed_docs)} new documents")
        
        # create embeddings and upload to Pinecone
        embedding_creator = EmbeddingCreator(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            openai_api_key=openai_api_key,
            batch_size=10  
        )
        
        embedding_creator.process_and_upload(chunks)
        print("Finished uploading to Pinecone")

if __name__ == "__main__":
    main()