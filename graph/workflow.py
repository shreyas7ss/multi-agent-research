# graph/workflow.py
"""
LangGraph Workflow - Orchestrates the multi-agent research pipeline.

Flow:
    START → Search → Analyzer → Synthesis → Reflection → END
                                     ↑          │
                                     └──────────┘ (if needs revision)
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
from agents.reflection import ReflectionAgent
from utils.logger import get_logger

console = Console()
logger = get_logger("workflow")


# Initialize agents
search_agent = WebSearchAgent()
analyzer_agent = DocumentAnalyzer()
synthesis_agent = SynthesisAgent()
reflection_agent = ReflectionAgent()


def search_node(state: ResearchState) -> ResearchState:
    """Search node - finds relevant web sources."""
    logger.info("Executing Search Node")
    return search_agent.run(state)


def analyzer_node(state: ResearchState) -> ResearchState:
    """Analyzer node - downloads, chunks, and stores documents."""
    logger.info("Executing Analyzer Node")
    
    # Convert sources to the format analyzer expects
    if state.sources and not state.approved_sources:
        # Auto-approve all sources for now (HITL can be added later)
        state.approved_sources = state.sources
    
    return analyzer_agent.run(state)


def synthesis_node(state: ResearchState) -> ResearchState:
    """Synthesis node - generates research report from stored chunks."""
    logger.info("Executing Synthesis Node")
    return synthesis_agent.run(state)


def reflection_node(state: ResearchState) -> ResearchState:
    """Reflection node - evaluates report quality."""
    logger.info("Executing Reflection Node")
    return reflection_agent.run(state)


def should_revise(state: ResearchState) -> Literal["revise", "end"]:
    """
    Conditional edge: decide whether to revise or end.
    """
    if state.error:
        logger.warning(f"Workflow ending due to error: {state.error}")
        return "end"
    
    if state.needs_revision and state.iteration_count < state.max_iterations:
        logger.info(f"Report needs revision (iteration {state.iteration_count})")
        return "revise"
    
    return "end"


def build_research_graph() -> StateGraph:
    """
    Build and return the research workflow graph.
    
    Flow: Search → Analyze → Synthesize → Reflect → (Revise or End)
    """
    logger.info("Building research workflow graph")
    
    # Create the graph with our state schema
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("search", search_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reflection", reflection_node)
    
    # Add edges
    graph.add_edge(START, "search")
    graph.add_edge("search", "analyzer")
    graph.add_edge("analyzer", "synthesis")
    graph.add_edge("synthesis", "reflection")
    
    # Conditional edge from reflection
    graph.add_conditional_edges(
        "reflection",
        should_revise,
        {
            "revise": "synthesis",  # Go back to synthesis with feedback
            "end": END
        }
    )
    
    logger.info("Research graph built with reflection loop")
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
        console.print(f"⭐ Quality score: {final_state.quality_score}/10")
        console.print(f"🔄 Iterations: {final_state.iteration_count}")
        
        if final_state.error:
            console.print(f"[red]⚠️ Error: {final_state.error}[/red]")
        
        logger.info(f"Research complete. Score: {final_state.quality_score}, Iterations: {final_state.iteration_count}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        console.print(f"[red]❌ Workflow failed: {e}[/red]")
        raise


def get_graph_visualization():
    """
    Get a Mermaid diagram representation of the graph.
    """
    return """
```mermaid
graph TD
    START((Start)) --> search[🔍 Search Agent]
    search --> analyzer[📄 Analyzer Agent]
    analyzer --> synthesis[📝 Synthesis Agent]
    synthesis --> reflection[🔍 Reflection Agent]
    reflection -->|Quality OK| END((End))
    reflection -->|Needs Revision| synthesis
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
