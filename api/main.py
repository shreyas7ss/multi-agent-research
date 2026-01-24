# api/main.py
"""
Simple FastAPI endpoint for the Multi-Agent Research Assistant.
Run with: python -m uvicorn api.main:app --reload
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
from uuid import uuid4

from graph.workflow import run_research
from graph.state import ResearchState
from utils.logger import get_logger
from agents.clarification import clarification_agent

logger = get_logger("api")

app = FastAPI(
    title="Multi-Agent Research API",
    description="Research assistant powered by AI agents",
    version="1.0.0"
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # Alternative dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store tasks in memory
tasks: Dict[str, dict] = {}
# Store clarification sessions
clarification_sessions: Dict[str, dict] = {}


# === Request/Response Models ===

class ResearchRequest(BaseModel):
    """Request body for research."""
    query: str


class ClarifyRequest(BaseModel):
    """Request to analyze a query and get clarifying questions."""
    query: str


class ClarifyResponseRequest(BaseModel):
    """Request to submit answers to clarifying questions."""
    session_id: str
    responses: List[str]


class ClarificationResponse(BaseModel):
    """Response with clarification analysis."""
    session_id: str
    needs_clarification: bool
    analysis: str
    questions: List[str]
    suggested_refined_query: str


class RefinedQueryResponse(BaseModel):
    """Response with the refined query."""
    refined_query: str
    original_query: str
    questions: List[str]
    responses: List[str]


# === Background Tasks ===

def run_research_task(task_id: str, query: str):
    """Background task that runs the research."""
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # Run the full research pipeline (skip clarification since it's already done)
        state = run_research(query, thread_id=task_id, skip_clarification=True)
        
        # Update task with results
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["report"] = state.draft_report
        tasks[task_id]["sources_found"] = len(state.sources)
        tasks[task_id]["quality_score"] = state.quality_score
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Task {task_id} completed!")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


# === Endpoints ===

@app.get("/")
def home():
    """API home."""
    return {"message": "Multi-Agent Research API", "docs": "/docs"}


@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}


# --- Clarification Endpoints ---

@app.post("/clarify", response_model=ClarificationResponse)
def analyze_query(request: ClarifyRequest):
    """
    Step 1: Analyze a query and get clarifying questions.
    
    Returns whether clarification is needed and a list of questions.
    Save the session_id to submit responses later.
    """
    try:
        # Analyze the query
        analysis = clarification_agent.analyze_query_api(request.query)
        
        # Create a session to store context
        session_id = str(uuid4())
        clarification_sessions[session_id] = {
            "original_query": request.query,
            "analysis": analysis,
            "created_at": datetime.now().isoformat()
        }
        
        return ClarificationResponse(
            session_id=session_id,
            needs_clarification=analysis["needs_clarification"],
            analysis=analysis["analysis"],
            questions=analysis["questions"],
            suggested_refined_query=analysis["suggested_refined_query"]
        )
        
    except Exception as e:
        logger.error(f"Clarification analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clarify/respond", response_model=RefinedQueryResponse)
def submit_clarification_responses(request: ClarifyResponseRequest):
    """
    Step 2: Submit answers to clarifying questions.
    
    Returns the refined query based on your answers.
    """
    session_id = request.session_id
    
    if session_id not in clarification_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Start with POST /clarify first.")
    
    session = clarification_sessions[session_id]
    original_query = session["original_query"]
    questions = session["analysis"]["questions"]
    
    # Validate response count
    if len(request.responses) != len(questions):
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {len(questions)} responses, got {len(request.responses)}"
        )
    
    try:
        # Refine the query
        refined_query = clarification_agent.refine_query_api(
            original_query=original_query,
            questions=questions,
            responses=request.responses
        )
        
        # Update session
        session["responses"] = request.responses
        session["refined_query"] = refined_query
        
        return RefinedQueryResponse(
            refined_query=refined_query,
            original_query=original_query,
            questions=questions,
            responses=request.responses
        )
        
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Research Endpoints ---

@app.post("/research")
def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a research task.
    
    Send a POST request with JSON body: {"query": "your question"}
    You can use a refined query from the clarification flow.
    """
    task_id = str(uuid4())
    
    tasks[task_id] = {
        "task_id": task_id,
        "query": request.query,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Run in background
    background_tasks.add_task(run_research_task, task_id, request.query)
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": f"Research started for: {request.query}"
    }


@app.get("/research/quick")
def quick_research(query: str, background_tasks: BackgroundTasks):
    """
    Quick way to start research using URL parameter.
    Example: /research/quick?query=quantum computing
    """
    task_id = str(uuid4())
    
    tasks[task_id] = {
        "task_id": task_id,
        "query": query,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(run_research_task, task_id, query)
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": f"Research started for: {query}"
    }
    

@app.get("/research/{task_id}")
def get_research(task_id: str):
    """Get research status and results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]
