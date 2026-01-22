# tests/test_synthesis.py
"""
Quick test script for the Synthesis Agent.
Run with: python -m tests.test_synthesis

Note: This requires data in the vector store. Run test_analyzer first!
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def test_synthesis():
    """Test the synthesis agent with data from vector store"""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Testing Synthesis Agent[/bold blue]\n"
        "This will retrieve chunks from Qdrant and generate a research report.\n"
        "[yellow]Note: Run test_analyzer first to populate the vector store![/yellow]",
        title="Test"
    ))
    
    # Import here to see initialization
    console.print("\n[yellow]Initializing synthesis agent...[/yellow]")
    
    from agents.synthesis import SynthesisAgent
    from graph.state import ResearchState
    
    agent = SynthesisAgent()
    console.print("[green]✓ Synthesis Agent initialized![/green]\n")
    
    # Create test state with a research query
    # This should match the data we stored in test_analyzer (quantum computing)
    state = ResearchState(
        original_query="What is quantum superposition and how does it work?"
    )
    
    console.print(f"[blue]Testing with query:[/blue] {state.original_query}\n")
    
    # Run the synthesis agent
    updated_state = agent.run(state)
    
    # Show results
    console.print(f"\n[bold]Results Summary:[/bold]")
    console.print(f"  - Chunks retrieved: {len(updated_state.retrieved_chunks)}")
    console.print(f"  - Report length: {len(updated_state.draft_report)} characters")
    
    if updated_state.error:
        console.print(f"  - [red]Error: {updated_state.error}[/red]")
    else:
        console.print("\n[bold green]✅ Synthesis test passed![/bold green]")
        
        # Save report to file
        report_path = "./docs/sample_report.md"
        import os
        os.makedirs("./docs", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Research Report\n\n")
            f.write(f"**Query:** {state.original_query}\n\n")
            f.write("---\n\n")
            f.write(updated_state.draft_report)
        
        console.print(f"\n[green]📄 Full report saved to: {report_path}[/green]")
    
    return updated_state


def test_synthesis_direct():
    """Test the direct synthesis method"""
    console.print("\n[bold blue]Testing direct synthesis...[/bold blue]")
    
    from agents.synthesis import synthesis_agent
    
    query = "What are the applications of quantum computing?"
    report = synthesis_agent.synthesize(query, top_k=10)
    
    console.print(Panel(
        Markdown(report[:2000] + "..." if len(report) > 2000 else report),
        title="Generated Report"
    ))
    
    console.print(f"\n[green]✓ Direct synthesis complete ({len(report)} chars)[/green]")


if __name__ == "__main__":
    test_synthesis()
