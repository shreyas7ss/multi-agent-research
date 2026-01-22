# agents/analyzer.py
"""
Document Analyzer Agent using LangChain components.

Responsibilities:
- Load documents from URLs (HTML, PDF)
- Split into chunks with configurable size/overlap
- Store chunks with embeddings in Qdrant vector database
"""

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional
import tempfile
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils.config import get_settings
from utils.vector_store import get_vector_store
from graph.state import Source, ResearchState

console = Console()
settings = get_settings()


class DocumentAnalyzer:
    """
    Analyzes, chunks, and stores documents using LangChain components.
    Compatible with LangGraph workflows.
    """
    
    def __init__(self):
        # Initialize text splitter with settings from config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def run(self, state: ResearchState) -> ResearchState:
        """
        Main entry point for LangGraph workflow.
        Processes all approved sources and stores in vector DB.
        
        Args:
            state: Current research state with sources to analyze
            
        Returns:
            Updated state with chunks_stored count
        """
        console.print("\n[bold blue]📄 Document Analyzer Agent Starting...[/bold blue]")
        
        # Use approved sources if available, otherwise all sources
        sources_to_process = state.approved_sources if state.approved_sources else state.sources
        
        if not sources_to_process:
            console.print("[yellow]⚠️ No sources to analyze[/yellow]")
            state.error = "No sources provided for analysis"
            return state
        
        all_chunks: List[Document] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(sources_to_process))
            
            for source in sources_to_process:
                try:
                    progress.update(task, description=f"Loading: {source.url[:50]}...")
                    
                    # Load document using appropriate loader
                    docs = self._load_document(source.url)
                    
                    if docs:
                        # Add source metadata to each document
                        for doc in docs:
                            doc.metadata.update({
                                "source_url": source.url,
                                "title": source.title,
                            })
                        
                        # Split into chunks
                        chunks = self.text_splitter.split_documents(docs)
                        
                        # Add chunk index metadata
                        for i, chunk in enumerate(chunks):
                            chunk.metadata["chunk_index"] = i
                            chunk.metadata["total_chunks"] = len(chunks)
                        
                        all_chunks.extend(chunks)
                        console.print(f"  [green]✓[/green] {source.title[:40]}... ({len(chunks)} chunks)")
                    else:
                        console.print(f"  [red]✗[/red] Failed to load: {source.url[:50]}")
                        
                except Exception as e:
                    console.print(f"  [red]✗[/red] Error: {source.url[:50]} - {str(e)}")
                
                progress.advance(task)
        
        # Store all chunks in vector database
        if all_chunks:
            console.print(f"\n[blue]Storing {len(all_chunks)} chunks in vector database...[/blue]")
            get_vector_store().add_documents(all_chunks)
        
        # Update state
        state.chunks_stored = len(all_chunks)
        state.analysis_complete = True
        state.current_agent = "analyzer"
        
        console.print(f"\n[bold green]✅ Analysis complete! Stored {len(all_chunks)} chunks[/bold green]")
        
        return state
    
    def _load_document(self, url: str) -> Optional[List[Document]]:
        """
        Load document from URL using appropriate LangChain loader.
        
        Args:
            url: URL to load
            
        Returns:
            List of LangChain Document objects, or None if failed
        """
        try:
            # Check if PDF
            if url.lower().endswith(".pdf"):
                return self._load_pdf(url)
            else:
                return self._load_web(url)
                
        except Exception as e:
            console.print(f"[red]Load error: {e}[/red]")
            return None
    
    def _load_web(self, url: str) -> List[Document]:
        """Load web page using WebBaseLoader"""
        loader = WebBaseLoader(
            web_paths=[url],
            header_template=self.headers
        )
        return loader.load()
    
    def _load_pdf(self, url: str) -> List[Document]:
        """Download PDF and load using PyPDFLoader"""
        # Download PDF to temp file
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Load using PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        return docs
    
    def analyze_urls(self, urls: List[str]) -> int:
        """
        Convenience method to analyze multiple URLs directly.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            Total number of chunks stored
        """
        all_chunks = []
        
        for url in urls:
            docs = self._load_document(url)
            if docs:
                for doc in docs:
                    doc.metadata["source_url"] = url
                
                chunks = self.text_splitter.split_documents(docs)
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["total_chunks"] = len(chunks)
                
                all_chunks.extend(chunks)
        
        if all_chunks:
            get_vector_store().add_documents(all_chunks)
        
        return len(all_chunks)
    
    def analyze_single_url(self, url: str, title: str = "Unknown") -> int:
        """
        Convenience method to analyze a single URL.
        
        Args:
            url: URL to analyze
            title: Title for the source
            
        Returns:
            Number of chunks stored
        """
        docs = self._load_document(url)
        if not docs:
            return 0
        
        for doc in docs:
            doc.metadata.update({
                "source_url": url,
                "title": title
            })
        
        chunks = self.text_splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        get_vector_store().add_documents(chunks)
        return len(chunks)


# Singleton instance
document_analyzer = DocumentAnalyzer()
