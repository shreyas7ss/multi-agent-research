# tests/test_workflow.py
"""
End-to-end test for the LangGraph research workflow.
Run with: python -m tests.test_workflow

This runs the FULL pipeline: Search → Analyze → Synthesize
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import os

console = Console()


def test_workflow():
    """Run the complete research workflow end-to-end."""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Testing Full Research Workflow[/bold blue]\n"
        "This will run: Search → Analyze → Synthesize\n"
        "[yellow]⏱️ This may take 2-3 minutes[/yellow]",
        title="End-to-End Test"
    ))
    
    # Import the workflow
    console.print("\n[yellow]Initializing workflow...[/yellow]")
    
    from graph.workflow import run_research
    
    console.print("[green]✓ Workflow initialized![/green]\n")
    
    # Research query
    query = "What are the latest breakthroughs in quantum computing for cryptography?"
    
    console.print(f"[cyan]Research Query:[/cyan] {query}\n")
    console.print("[blue]Starting full research pipeline...[/blue]\n")
    
    # Run the workflow
    final_state = run_research(query, thread_id="test_run_1")
    
    # Display results summary
    console.print("\n" + "=" * 60)
    console.print("[bold]📊 Final Results:[/bold]")
    console.print(f"  • Search queries generated: {len(final_state.search_queries)}")
    console.print(f"  • Sources found: {len(final_state.sources)}")
    console.print(f"  • Chunks stored: {final_state.chunks_stored}")
    console.print(f"  • Report length: {len(final_state.draft_report)} chars")
    
    if final_state.source_diversity:
        console.print("\n[bold]📚 Source Diversity:[/bold]")
        for stype, count in final_state.source_diversity.items():
            console.print(f"  • {stype}: {count}")
    
    # Save report to file
    os.makedirs("./docs", exist_ok=True)
    report_path = "./docs/full_pipeline_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"**Sources Analyzed:** {len(final_state.sources)}\n\n")
        f.write("---\n\n")
        f.write(final_state.draft_report)
    
    console.print(f"\n[green]📄 Full report saved to: {report_path}[/green]")
    
    # Show report preview
    if final_state.draft_report:
        console.print("\n[bold]📝 Report Preview:[/bold]")
        preview = final_state.draft_report[:1000] + "..." if len(final_state.draft_report) > 1000 else final_state.draft_report
        console.print(Panel(
            Markdown(preview),
            title="Report Preview",
            border_style="green"
        ))
    
    console.print("\n[bold green]✅ Workflow test complete![/bold green]")
    
    return final_state


def test_workflow_simple():
    """Quick test with a simpler query."""
    from graph.workflow import run_research
    
    query = "What is machine learning?"
    state = run_research(query, thread_id="simple_test")
    
    print(f"\nReport length: {len(state.draft_report)} chars")
    return state


if __name__ == "__main__":
    test_workflow()
