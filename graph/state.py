# graph/state.py
"""
Shared state definitions for the multi-agent research workflow.
All agents read from and write to this state.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from langchain_core.messages import BaseMessage
from datetime import datetime


class Source(BaseModel):
    """Represents a discovered web source with rich metadata"""
    url: str
    title: str
    snippet: str
    approved: bool = False  # HITL approval status
    
    # Enhanced metadata for better citations
    author: Optional[str] = None
    publication: Optional[str] = None  # e.g., "Nature", "TechCrunch"
    published_date: Optional[str] = None  # ISO format date string
    source_type: Literal["academic", "news", "industry", "blog", "wiki", "other"] = "other"
    
    def get_citation(self, num: int) -> str:
        """Generate a formatted citation string"""
        parts = []
        if self.author:
            parts.append(self.author)
        if self.published_date:
            try:
                year = self.published_date[:4]
                parts.append(f"({year})")
            except:
                pass
        parts.append(f'"{self.title}"')
        if self.publication:
            parts.append(self.publication)
        parts.append(self.url)
        
        return f"[{num}] " + ". ".join(parts)


class DocumentChunk(BaseModel):
    """Represents a retrieved document chunk"""
    text: str
    score: float
    source_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Enhanced metadata
    source_title: Optional[str] = None
    source_type: Optional[str] = None
    published_date: Optional[str] = None


class ResearchState(BaseModel):
    """
    Shared state that flows through all agents in the research workflow.
    Each agent reads from and writes to this state.
    """
    
    # ============ USER INPUT ============
    original_query: str = ""
    refined_query: str = ""
    
    # ============ CLARIFICATION AGENT ============
    clarification_questions: List[str] = Field(default_factory=list)
    user_responses: List[str] = Field(default_factory=list)
    clarification_complete: bool = False
    
    # ============ SEARCH AGENT ============
    search_queries: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    approved_sources: List[Source] = Field(default_factory=list)  # After HITL
    
    # Source diversity tracking
    source_diversity: Dict[str, int] = Field(default_factory=dict)  # {"academic": 5, "news": 3, ...}
    
    # ============ ANALYZER AGENT ============
    chunks_stored: int = 0
    analysis_complete: bool = False
    
    # ============ SYNTHESIS AGENT ============
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    draft_report: str = ""
    
    # ============ REFLECTION AGENT ============
    quality_score: float = 0.0
    needs_revision: bool = False
    revision_feedback: str = ""
    iteration_count: int = 0
    max_iterations: int = 3
    
    # ============ FINAL OUTPUT ============
    final_report: str = ""
    
    # ============ CONVERSATION (for HITL) ============
    messages: List[Any] = Field(default_factory=list)  # BaseMessage objects
    
    # ============ WORKFLOW CONTROL ============
    current_agent: str = ""
    error: Optional[str] = None
    
    # ============ METADATA ============
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        arbitrary_types_allowed = True  # Allow BaseMessage type


def get_temporal_context(date_str: str) -> str:
    """
    Add relative time context to a date string.
    
    Args:
        date_str: ISO format date string (e.g., "2024-03-15")
        
    Returns:
        Human-readable relative time (e.g., "Last year (2024)")
    """
    try:
        if not date_str:
            return ""
            
        event_date = datetime.fromisoformat(date_str[:10])
        current_year = datetime.now().year
        event_year = event_date.year
        years_ago = current_year - event_year
        
        if years_ago == 0:
            return f"This year ({event_year})"
        elif years_ago == 1:
            return f"Last year ({event_year})"
        elif years_ago < 3:
            return f"Recently ({event_year})"
        elif years_ago < 5:
            return f"{years_ago} years ago ({event_year})"
        else:
            return f"In {event_year}"
    except:
        return date_str
