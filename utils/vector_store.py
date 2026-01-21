# utils/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import os
from utils.config import get_settings

settings = get_settings()

class VectorStore:
    """Manages Qdrant vector database operations"""
    
    def __init__(self):
        # Use local storage by default
        storage_path = "./storage/qdrant_storage"
        os.makedirs(storage_path, exist_ok=True)
        
        if settings.qdrant_url:
            # Use Qdrant Cloud if configured
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            # Use local Qdrant
            self.client = QdrantClient(path=storage_path)
        
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        self.collection_name = settings.vector_db_collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.vector_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {self.collection_name}")
        else:
            print(f"✅ Collection already exists: {self.collection_name}")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]) -> None:
        """Add documents to vector store"""
        # Generate embeddings
        vectors = self.embeddings.embed_documents(texts)
        
        # Create points
        points = [
            PointStruct(
                id=i,
                vector=vector,
                payload={"text": text, **metadata}
            )
            for i, (text, vector, metadata) in enumerate(zip(texts, vectors, metadatas))
        ]
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Search for similar documents"""
        query_vector = self.embeddings.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            {
                "text": result.payload["text"],
                "score": result.score,
                **{k: v for k, v in result.payload.items() if k != "text"}
            }
            for result in results
        ]

# Singleton instance
vector_store = VectorStore()