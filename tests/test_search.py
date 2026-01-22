# tests/test_search.py
"""
Quick test script for the Web Search Agent.
Run with: python -m tests.test_search
"""

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_search():
    """Test the web search agent with a sample query"""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Testing Web Search Agent[/bold blue]\n"
        "This will generate search queries and find sources using Tavily.",
        title="Test"
    ))
    
    # Import here to see the initialization progress
    console.print("\n[yellow]Initializing search agent...[/yellow]")
    
    from agents.search import WebSearchAgent
    from graph.state import ResearchState
    
    # Initialize
    agent = WebSearchAgent()
    console.print("[green]✓ Search Agent initialized![/green]\n")
    
    # Create test state with a research query
    state = ResearchState(
        original_query="What are the latest developments in quantum computing for drug discovery?"
    )
    
    console.print(f"[blue]Testing with query:[/blue] {state.original_query}\n")
    
    # Run the search agent
    updated_state = agent.run(state)
    
    # Show results summary
    console.print(f"\n[bold]Results Summary:[/bold]")
    console.print(f"  - Search queries generated: {len(updated_state.search_queries)}")
    console.print(f"  - Sources found: {len(updated_state.sources)}")
    
    if updated_state.sources:
        console.print(f"\n[bold]Sample Source:[/bold]")
        source = updated_state.sources[0]
        console.print(f"  Title: {source.title}")
        console.print(f"  URL: {source.url}")
        console.print(f"  Snippet: {source.snippet[:200]}...")
    
    console.print("\n[bold green]✅ Search test passed![/bold green]")
    
    return updated_state


if __name__ == "__main__":
    test_search()
