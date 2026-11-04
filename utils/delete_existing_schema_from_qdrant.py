from qdrant_client import QdrantClient

client = QdrantClient(path="data/qdrant_db")

collection_name = "enterprise_rag_chunks"

if client.collection_exists(collection_name):
    
    client.delete_collection(collection_name)
    print(f"Old Collection deleted")

client.close()