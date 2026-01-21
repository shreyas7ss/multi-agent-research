# graph/state.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from langchain_core.messages import BaseMessage


class Source(BaseModel):
    """Represents a discovered web source"""
    url: str
    title: str
    snippet: str
    approved: bool = False  # HITL approval status


class DocumentChunk(BaseModel):
    """Represents a retrieved document chunk"""
    text: str
    score: float
    source_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    
    class Config:
        arbitrary_types_allowed = True  # Allow BaseMessage type
