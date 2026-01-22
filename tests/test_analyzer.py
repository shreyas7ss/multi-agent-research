# tests/test_analyzer.py
"""
Quick test script for the Document Analyzer agent.
Run with: python -m tests.test_analyzer
"""

from rich.console import Console
from rich.panel import Panel

console = Console()

def test_analyzer():
    """Test the document analyzer with a sample URL"""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Testing Document Analyzer[/bold blue]\n"
        "This will download a sample page, chunk it, and store in Qdrant.",
        title="Test"
    ))
    
    # Import here to see the model loading progress
    console.print("\n[yellow]Loading embedding model (first time may take ~1 min)...[/yellow]")
    
    from agents.analyzer import DocumentAnalyzer
    from utils.vector_store import get_vector_store
    
    # Initialize
    analyzer = DocumentAnalyzer()
    vector_store = get_vector_store()
    
    console.print("[green]✓ Analyzer and Vector Store initialized![/green]\n")
    
    # Test with a sample Wikipedia page
    test_url = "https://en.wikipedia.org/wiki/Quantum_computing"
    test_title = "Quantum Computing - Wikipedia"
    
    console.print(f"[blue]Analyzing: {test_url}[/blue]")
    
    chunks_stored = analyzer.analyze_single_url(test_url, test_title)
    
    console.print(f"\n[bold green]✅ Success! Stored {chunks_stored} chunks[/bold green]")
    
    # Test search
    console.print("\n[blue]Testing search...[/blue]")
    query = "What is quantum superposition?"
    results = vector_store.search(query, top_k=3)
    
    console.print(f"\n[bold]Search: '{query}'[/bold]")
    console.print(f"Found {len(results)} results:\n")
    
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("similarity_score", "N/A")
        preview = doc.page_content[:200].replace("\n", " ")
        console.print(f"[cyan]{i}.[/cyan] (score: {score:.4f})")
        console.print(f"   {preview}...\n")
    
    console.print("[bold green]✅ All tests passed![/bold green]")


if __name__ == "__main__":
    test_analyzer()
