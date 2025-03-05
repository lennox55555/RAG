from pinecone import Pinecone

# init 
pc = Pinecone(api_key="pcsk_41GHre_JdhTRFid6dboFnFuBow2PxfVsV2iG2F38hTmqRfbizfjv3Znj2eXx3BvtUP6ywp")
index = pc.Index("mff")


vector_1536 = [0.01] * 1536 

vectors = [
    {
        "id": "vec1",
        "values": vector_1536,
        "metadata": {"genre": "drama"}
    },
    {
        "id": "vec2",
        "values": vector_1536, 
        "metadata": {"genre": "action"}
    }
]

try:
    # Upsert to index
    index.upsert(
        vectors=vectors,
        namespace="ns1"
    )
    print("Successfully uploaded vectors to index")

except Exception as e:
    print(f"Error: {str(e)}")