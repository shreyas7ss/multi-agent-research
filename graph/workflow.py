# graph/workflow.py
"""
LangGraph Workflow - Orchestrates the multi-agent research pipeline.

Flow:
    START → Search → Analyzer → Synthesis → END
    
Future Flow (with all agents):
    START → Clarification → Search → Analyzer → Synthesis → Reflection → END
                                                              ↓
                                                        (if needs revision)
                                                              ↓
                                                        Back to Synthesis
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from rich.console import Console
from rich.panel import Panel

from graph.state import ResearchState
from agents.search import WebSearchAgent
from agents.analyzer import DocumentAnalyzer
from agents.synthesis import SynthesisAgent
from utils.logger import get_logger

console = Console()
logger = get_logger("workflow")


# Initialize agents
search_agent = WebSearchAgent()
analyzer_agent = DocumentAnalyzer()
synthesis_agent = SynthesisAgent()


def search_node(state: ResearchState) -> ResearchState:
    """Search node - finds relevant web sources."""
    logger.info("Executing Search Node")
    return search_agent.run(state)


def analyzer_node(state: ResearchState) -> ResearchState:
    """Analyzer node - downloads, chunks, and stores documents."""
    logger.info("Executing Analyzer Node")
    
    # Convert sources to the format analyzer expects
    # Analyzer will process all sources found by search
    if state.sources and not state.approved_sources:
        # Auto-approve all sources for now (HITL can be added later)
        state.approved_sources = state.sources
    
    return analyzer_agent.run(state)


def synthesis_node(state: ResearchState) -> ResearchState:
    """Synthesis node - generates research report from stored chunks."""
    logger.info("Executing Synthesis Node")
    return synthesis_agent.run(state)


def should_continue(state: ResearchState) -> Literal["continue", "end"]:
    """
    Conditional edge: decide whether to continue or end.
    Currently always ends, but can be extended for reflection loop.
    """
    if state.error:
        logger.warning(f"Workflow ending due to error: {state.error}")
        return "end"
    
    # Future: Check reflection agent's quality score
    # if state.needs_revision and state.iteration_count < state.max_iterations:
    #     return "revise"
    
    return "end"


def build_research_graph() -> StateGraph:
    """
    Build and return the research workflow graph.
    
    Current flow: Search → Analyze → Synthesize
    """
    logger.info("Building research workflow graph")
    
    # Create the graph with our state schema
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("search", search_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("synthesis", synthesis_node)
    
    # Add edges (linear flow for now)
    graph.add_edge(START, "search")
    graph.add_edge("search", "analyzer")
    graph.add_edge("analyzer", "synthesis")
    graph.add_edge("synthesis", END)
    
    logger.info("Research graph built successfully")
    return graph


def create_research_app(checkpointer=None):
    """
    Create and compile the research application.
    
    Args:
        checkpointer: Optional memory saver for conversation persistence
        
    Returns:
        Compiled LangGraph application
    """
    graph = build_research_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    app = graph.compile(checkpointer=checkpointer)
    logger.info("Research app compiled successfully")
    
    return app


def run_research(query: str, thread_id: str = "default") -> ResearchState:
    """
    Run the complete research pipeline.
    
    Args:
        query: The research question
        thread_id: Unique thread ID for conversation memory
        
    Returns:
        Final ResearchState with the generated report
    """
    console.print(Panel.fit(
        f"[bold blue]🔬 Multi-Agent Research Assistant[/bold blue]\n\n"
        f"[cyan]Query:[/cyan] {query}",
        title="Starting Research"
    ))
    
    logger.info(f"Starting research for query: {query}")
    
    # Create the app
    app = create_research_app()
    
    # Initial state
    initial_state = ResearchState(original_query=query)
    
    # Config for this thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph
    console.print("\n[bold]Running research pipeline...[/bold]\n")
    
    try:
        # Execute the workflow
        final_state = app.invoke(initial_state, config)
        
        # Convert dict back to ResearchState if needed
        if isinstance(final_state, dict):
            final_state = ResearchState(**final_state)
        
        # Display results
        console.print("\n" + "=" * 60)
        console.print("[bold green]✅ Research Complete![/bold green]")
        console.print(f"📊 Sources found: {len(final_state.sources)}")
        console.print(f"📚 Chunks stored: {final_state.chunks_stored}")
        console.print(f"📄 Report length: {len(final_state.draft_report)} characters")
        
        if final_state.error:
            console.print(f"[red]⚠️ Error: {final_state.error}[/red]")
        
        logger.info(f"Research complete. Report: {len(final_state.draft_report)} chars")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        console.print(f"[red]❌ Workflow failed: {e}[/red]")
        raise


def get_graph_visualization():
    """
    Get a Mermaid diagram representation of the graph.
    Useful for documentation.
    """
    return """
```mermaid
graph TD
    START((Start)) --> search[🔍 Search Agent]
    search --> analyzer[📄 Analyzer Agent]
    analyzer --> synthesis[📝 Synthesis Agent]
    synthesis --> END((End))
    
    search -->|"Finds 20+ sources"| analyzer
    analyzer -->|"Chunks & stores"| synthesis
    synthesis -->|"Generates report"| END
```
"""


# Expose the main app
research_app = None

def get_research_app():
    """Get or create the research app singleton."""
    global research_app
    if research_app is None:
        research_app = create_research_app()
    return research_app
