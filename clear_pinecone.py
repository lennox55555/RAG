# clear_pinecone.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("mff")

# delete all vectors in the 'documents' namespace
index.delete(delete_all=True, namespace="documents")
print("Cleared all vectors from the 'mff' index in the 'documents' namespace.")

# verify the index is empty
stats = index.describe_index_stats()
print("Index stats after clearing:", stats)
