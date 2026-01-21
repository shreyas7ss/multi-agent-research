# utils/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str
    anthropic_api_key: str
    tavily_api_key: str
    
    # Optional Qdrant Cloud
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    
    # Project Settings
    project_name: str = "multi-agent-research"
    vector_db_collection: str = "research_documents"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "claude-sonnet-4-20250514"
    
    # Vector DB Settings
    vector_dimension: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Search Settings
    max_search_results: int = 5
    num_search_queries: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return Settings()