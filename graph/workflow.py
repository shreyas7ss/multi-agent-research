# graph/workflow.py
"""
LangGraph Workflow - Orchestrates the multi-agent research pipeline.

Full Flow:
    START → Clarification → Search → Analyzer → Synthesis → Reflection → END
                                                     ↑          │
                                                     └──────────┘ (if needs revision)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from rich.console import Console
from rich.panel import Panel

from graph.state import ResearchState
from agents.clarification import ClarificationAgent
from agents.search import WebSearchAgent
from agents.analyzer import DocumentAnalyzer
from agents.synthesis import SynthesisAgent
from agents.reflection import ReflectionAgent
from utils.logger import get_logger

console = Console()
logger = get_logger("workflow")


# Initialize agents
clarification_agent = ClarificationAgent()
search_agent = WebSearchAgent()
analyzer_agent = DocumentAnalyzer()
synthesis_agent = SynthesisAgent()
reflection_agent = ReflectionAgent()


def clarification_node(state: ResearchState) -> ResearchState:
    """Clarification node - refines user query."""
    logger.info("Executing Clarification Node")
    return clarification_agent.run(state)


def search_node(state: ResearchState) -> ResearchState:
    """Search node - finds relevant web sources."""
    logger.info("Executing Search Node")
    return search_agent.run(state)


def analyzer_node(state: ResearchState) -> ResearchState:
    """Analyzer node - downloads, chunks, and stores documents."""
    logger.info("Executing Analyzer Node")
    
    if state.sources and not state.approved_sources:
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
    """Conditional edge: decide whether to revise or end."""
    if state.error:
        logger.warning(f"Workflow ending due to error: {state.error}")
        return "end"
    
    if state.needs_revision and state.iteration_count < state.max_iterations:
        logger.info(f"Report needs revision (iteration {state.iteration_count})")
        return "revise"
    
    return "end"


def build_research_graph(include_clarification: bool = True) -> StateGraph:
    """
    Build and return the research workflow graph.
    
    Args:
        include_clarification: Whether to include the clarification step
        
    Flow: Clarification → Search → Analyze → Synthesize → Reflect → (Revise or End)
    """
    logger.info("Building research workflow graph")
    
    graph = StateGraph(ResearchState)
    
    # Add nodes
    if include_clarification:
        graph.add_node("clarification", clarification_node)
    graph.add_node("search", search_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reflection", reflection_node)
    
    # Add edges
    if include_clarification:
        graph.add_edge(START, "clarification")
        graph.add_edge("clarification", "search")
    else:
        graph.add_edge(START, "search")
    
    graph.add_edge("search", "analyzer")
    graph.add_edge("analyzer", "synthesis")
    graph.add_edge("synthesis", "reflection")
    
    # Conditional edge from reflection
    graph.add_conditional_edges(
        "reflection",
        should_revise,
        {
            "revise": "synthesis",
            "end": END
        }
    )
    
    logger.info("Research graph built with all agents")
    return graph


def create_research_app(checkpointer=None, include_clarification: bool = True):
    """
    Create and compile the research application.
    
    Args:
        checkpointer: Optional memory saver for conversation persistence
        include_clarification: Whether to include clarification step
        
    Returns:
        Compiled LangGraph application
    """
    graph = build_research_graph(include_clarification=include_clarification)
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    app = graph.compile(checkpointer=checkpointer)
    logger.info("Research app compiled successfully")
    
    return app


def run_research(
    query: str, 
    thread_id: str = "default",
    skip_clarification: bool = False
) -> ResearchState:
    """
    Run the complete research pipeline.
    
    Args:
        query: The research question
        thread_id: Unique thread ID for conversation memory
        skip_clarification: Set True to skip clarification step
        
    Returns:
        Final ResearchState with the generated report
    """
    mode = "with clarification" if not skip_clarification else "skip clarification"
    
    console.print(Panel.fit(
        f"[bold blue]🔬 Multi-Agent Research Assistant[/bold blue]\n\n"
        f"[cyan]Query:[/cyan] {query}\n"
        f"[dim]Mode: {mode}[/dim]",
        title="Starting Research"
    ))
    
    logger.info(f"Starting research for query: {query}")
    
    # Create the app
    app = create_research_app(include_clarification=not skip_clarification)
    
    # Initial state
    initial_state = ResearchState(original_query=query)
    
    # Config for this thread
    config = {"configurable": {"thread_id": thread_id}}
    
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
        
        if final_state.refined_query and final_state.refined_query != query:
            console.print(f"✨ Refined query: {final_state.refined_query}")
        
        if final_state.error:
            console.print(f"[red]⚠️ Error: {final_state.error}[/red]")
        
        logger.info(f"Research complete. Score: {final_state.quality_score}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        console.print(f"[red]❌ Workflow failed: {e}[/red]")
        raise


def get_graph_visualization():
    """Get a Mermaid diagram representation of the graph."""
    return """
```mermaid
graph TD
    START((Start)) --> clarification[💬 Clarification]
    clarification --> search[🔍 Search Agent]
    search --> analyzer[📄 Analyzer Agent]
    analyzer --> synthesis[📝 Synthesis Agent]
    synthesis --> reflection[🔍 Reflection Agent]
    reflection -->|Quality OK| END((End))
    reflection -->|Needs Revision| synthesis
```
"""


# Expose the main app
research_app = None

def get_research_app(include_clarification: bool = True):
    """Get or create the research app singleton."""
    global research_app
    if research_app is None:
        research_app = create_research_app(include_clarification=include_clarification)
    return research_app
