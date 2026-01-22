# utils/vector_store.py
"""
Vector Store using LangChain's Qdrant integration.
Uses HuggingFace BGE embeddings for semantic search.
"""

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import List, Dict, Optional
import os

from utils.config import get_settings

settings = get_settings()


class VectorStore:
    """Manages Qdrant vector database operations using LangChain"""
    
    def __init__(self):
        # Setup storage path
        self.storage_path = "./storage/qdrant_storage"
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize HuggingFace embeddings with BGE model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.collection_name = settings.vector_db_collection
        self.vector_dimension = settings.vector_dimension
        
        # Initialize Qdrant client
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            self.client = QdrantClient(path=self.storage_path)
        
        # Ensure collection exists
        self._initialize_collection()
        
        # Create LangChain vector store
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
    
    def _initialize_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add LangChain Document objects to vector store.
        
        Args:
            documents: List of LangChain Document objects with page_content and metadata
        """
        if documents:
            self.vectorstore.add_documents(documents)
            print(f"✅ Added {len(documents)} documents to vector store")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        Add raw texts to vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
        """
        if texts:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            print(f"✅ Added {len(texts)} texts to vector store")
    
    def search(self, query: str, top_k: int = 20) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of LangChain Document objects with similarity scores
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Add score to document metadata
        documents = []
        for doc, score in results:
            doc.metadata["similarity_score"] = score
            documents.append(doc)
        
        return documents
    
    def search_with_scores(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        Search and return documents with similarity scores.
        
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=top_k)
    
    def get_retriever(self, top_k: int = 20):
        """
        Get a LangChain retriever for use in chains.
        
        Returns:
            VectorStoreRetriever compatible with LangChain/LangGraph
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    def clear_collection(self):
        """Delete all documents in the collection"""
        self.client.delete_collection(self.collection_name)
        self._initialize_collection()
        print(f"🗑️ Cleared collection: {self.collection_name}")


# Lazy initialization to avoid loading model at import time
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store