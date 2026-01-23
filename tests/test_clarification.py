# tests/test_clarification.py
"""
Test script for the Clarification Agent.
Run with: python -m tests.test_clarification
"""

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_clarification():
    """Test the clarification agent interactively."""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Testing Clarification Agent[/bold blue]\n"
        "This will analyze a query and ask clarifying questions.",
        title="Test"
    ))
    
    console.print("\n[yellow]Initializing clarification agent...[/yellow]")
    
    from agents.clarification import ClarificationAgent
    from graph.state import ResearchState
    
    agent = ClarificationAgent()
    console.print("[green]✓ Clarification Agent initialized![/green]\n")
    
    # Test with an ambiguous query
    query = "Tell me about AI"  # Intentionally vague
    
    console.print(f"[cyan]Testing with vague query:[/cyan] \"{query}\"\n")
    
    state = ResearchState(original_query=query)
    
    # Run the agent (this will prompt for user input)
    updated_state = agent.run(state)
    
    # Show results
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Original: {updated_state.original_query}")
    console.print(f"  Refined:  {updated_state.refined_query}")
    console.print(f"  Questions asked: {len(updated_state.clarification_questions)}")
    
    console.print("\n[bold green]✅ Clarification test complete![/bold green]")
    
    return updated_state


def test_clear_query():
    """Test with a query that doesn't need clarification."""
    
    console.print("\n[bold blue]Testing with a clear query...[/bold blue]\n")
    
    from agents.clarification import ClarificationAgent
    from graph.state import ResearchState
    
    agent = ClarificationAgent()
    
    # Clear, specific query
    query = "What are the latest quantum computing breakthroughs in 2025 for drug discovery applications?"
    
    console.print(f"[cyan]Query:[/cyan] \"{query}\"\n")
    
    state = ResearchState(original_query=query)
    updated_state = agent.run(state)
    
    console.print(f"\n[green]Refined Query:[/green] {updated_state.refined_query}")
    console.print(f"[green]Needed clarification:[/green] {len(updated_state.clarification_questions) > 0}")


if __name__ == "__main__":
    test_clarification()
