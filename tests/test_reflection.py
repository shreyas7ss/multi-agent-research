# tests/test_reflection.py
"""
Test script for the Reflection Agent.
Run with: python -m tests.test_reflection
"""

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_reflection():
    """Test the reflection agent with a sample report."""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Testing Reflection Agent[/bold blue]\n"
        "This will evaluate a sample report and show quality scores.",
        title="Test"
    ))
    
    console.print("\n[yellow]Initializing reflection agent...[/yellow]")
    
    from agents.reflection import ReflectionAgent
    
    agent = ReflectionAgent()
    console.print("[green]✓ Reflection Agent initialized![/green]\n")
    
    # Sample query and report for testing
    query = "What are the latest developments in quantum computing?"
    
    sample_report = """
# Executive Summary

Quantum computing has seen significant developments in recent years. Companies like IBM, Google, and 
newcomers like IonQ are pushing the boundaries of what's possible. This report examines the current 
state of quantum computing, recent breakthroughs, and future directions.

## Key Findings

1. **Error correction advances** - IBM achieved 127-qubit processors in 2024 [1]
2. **Cloud access expands** - Major providers now offer quantum computing services
3. **Practical applications emerge** - Finance and drug discovery lead adoption

## Recent Developments

In 2024, several major announcements were made:
- IBM unveiled their Condor processor with 1,121 qubits
- Google claimed quantum advantage for specific problems
- Startups raised record funding for quantum hardware

## Challenges

- Decoherence remains a major issue
- Error rates are still too high for many applications
- Skilled workforce shortage

## Sources
[1] IBM Research. "Quantum Roadmap 2024". https://ibm.com/quantum
[2] Google AI. "Quantum Supremacy Update". https://ai.google/research
"""
    
    console.print("[blue]Evaluating sample report...[/blue]\n")
    
    # Evaluate
    score, needs_revision, feedback = agent.evaluate(query, sample_report)
    
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Score: {score}/10")
    console.print(f"  Needs Revision: {needs_revision}")
    
    if feedback:
        console.print(f"\n[bold]Feedback:[/bold]\n{feedback}")
    
    console.print("\n[bold green]✅ Reflection test complete![/bold green]")
    
    return score, needs_revision, feedback


if __name__ == "__main__":
    test_reflection()
