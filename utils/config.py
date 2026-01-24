# utils/config.py
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    groq_api_key: str
    tavily_api_key: str
    huggingface_api_key: str
    
    # LangSmith Monitoring
    langchain_api_key: str
    langchain_tracing_v2: bool = True
    langchain_project: str = "multi-agent-research"
    
    # Optional Qdrant Cloud
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    
    # Project Settings
    project_name: str = "multi-agent-research"
    vector_db_collection: str = "research_documents"
    
    # Embedding Model (BGE is excellent for research/RAG)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    vector_dimension: int = 768  # BGE-base dimension
    
    # LLM Model (Groq)
    llm_model: str = "llama-3.3-70b-versatile"
    
    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Search Settings
    max_search_results: int = 20
    num_search_queries: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings():
    """Get cached settings instance"""
    settings = Settings()
    
    # Set LangSmith environment variables so LangChain can detect them
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    
    return settings
