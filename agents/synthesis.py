# agents/synthesis.py
"""
Synthesis Agent - Generates comprehensive research reports with critical analysis.

Responsibilities:
- Retrieve relevant chunks from vector database
- Synthesize information from diverse sources
- Generate structured reports with proper citations
- Include critical analysis distinguishing hype from reality
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime

from utils.config import get_settings
from utils.vector_store import get_vector_store
from utils.logger import get_agent_logger
from graph.state import ResearchState, DocumentChunk, get_temporal_context

console = Console()
settings = get_settings()
logger = get_agent_logger("synthesis")


# Enhanced synthesis prompt with critical analysis requirements
SYNTHESIS_PROMPT = """You are a PhD-level research analyst tasked with synthesizing information into a comprehensive, CRITICALLY ANALYZED research report.

## Research Query
{query}

## Retrieved Information
{context}

## Report Requirements

Generate a research report with the following sections:

### 1. Executive Summary (3 paragraphs)
- High-level overview of key findings
- Main insights and conclusions
- Brief mention of limitations or debates

### 2. Key Findings (5-7 bullet points)
- Each finding must have a citation [1], [2], etc.
- Distinguish between established facts and emerging claims
- Note level of evidence (demonstrated vs. theoretical)

### 3. Detailed Analysis
Synthesize information across sources while:
- **Noting conflicting viewpoints**: "While Source A claims X, Source B argues Y..."
- **Distinguishing hype from reality**: "Despite optimistic claims, actual deployment remains limited..."
- **Identifying consensus vs. debate**: "Researchers broadly agree on X, but opinions differ on Y..."
- **Providing temporal context**: Use phrases like "As of 2024" or "In recent years"

### 4. Recent Developments
- Focus on developments from the past 2 years
- Include dates when available (e.g., "In March 2024...")
- Note which developments are confirmed vs. announced/planned

### 5. Challenges and Limitations
- Technical challenges mentioned by sources
- Practical implementation barriers
- Expert skepticism where relevant
- Knowledge gaps in current research

### 6. Future Directions
- Near-term expectations (1-2 years)
- Long-term possibilities (5+ years)
- Distinguish predictions from speculation
- Note expert disagreements on timelines

### 7. Sources
Format each source as:
[1] Author/Publication. "Title" (Year). URL

## Critical Analysis Guidelines
1. **Separate proven results from speculation** - Use phrases like "Lab demonstrations show... while commercial applications remain..."
2. **Include expert skepticism** - If sources disagree, represent both views
3. **Be specific about evidence** - "A 2024 study found..." vs "Studies suggest..."
4. **Avoid overgeneralizing** - Don't state things as facts unless sources confirm them
5. **Note source quality** - Academic papers vs. company announcements vs. news reports

## Formatting
- Use Markdown formatting
- Citations as [1], [2], etc. inline
- Target length: 2000-3000 words
- Academic tone but accessible to educated non-specialists

Generate the research report now:"""


class SynthesisAgent:
    """
    Synthesizes research reports from vector store chunks.
    Uses RAG with critical analysis layer.
    """
    
    def __init__(self):
        logger.info("Initializing Synthesis Agent")
        
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=4096
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a world-class research analyst with expertise in synthesizing complex information and providing critical analysis. You distinguish between established facts, emerging claims, and speculation."),
            ("human", SYNTHESIS_PROMPT)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        logger.info(f"Synthesis Agent initialized with model={settings.llm_model}")
    
    def run(self, state: ResearchState) -> ResearchState:
        """Main entry point for LangGraph workflow."""
        logger.info("Synthesis Agent starting")
        console.print("\n[bold blue]📝 Synthesis Agent Starting...[/bold blue]")
        
        query = state.refined_query if state.refined_query else state.original_query
        
        if not query:
            logger.error("No query provided for synthesis")
            state.error = "No query provided for synthesis"
            return state
        
        console.print(f"[cyan]Query:[/cyan] {query}\n")
        
        # Retrieve relevant chunks
        console.print("[blue]Retrieving relevant information from knowledge base...[/blue]")
        chunks = self._retrieve_chunks(query, top_k=settings.max_search_results)
        
        if not chunks:
            logger.warning("No relevant chunks found in vector store")
            console.print("[yellow]⚠️ No relevant information found. Try running the analyzer first.[/yellow]")
            state.error = "No relevant information found in knowledge base"
            return state
        
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        console.print(f"[green]✓ Retrieved {len(chunks)} relevant chunks[/green]")
        
        # Show source diversity
        source_types = {}
        for doc in chunks:
            stype = doc.metadata.get("source_type", "other")
            source_types[stype] = source_types.get(stype, 0) + 1
        
        if source_types:
            console.print("[dim]Source types: " + ", ".join(f"{k}({v})" for k, v in source_types.items()) + "[/dim]")
        
        # Store retrieved chunks in state
        state.retrieved_chunks = [
            DocumentChunk(
                text=doc.page_content,
                score=doc.metadata.get("similarity_score", 0.0),
                source_url=doc.metadata.get("source_url", ""),
                source_title=doc.metadata.get("title", ""),
                source_type=doc.metadata.get("source_type"),
                metadata=doc.metadata
            )
            for doc in chunks
        ]
        
        # Format context with enhanced metadata
        context = self._format_context(chunks)
        
        # Generate report
        console.print("\n[blue]Generating research report with critical analysis...[/blue]")
        console.print("[dim](This may take 30-60 seconds)[/dim]\n")
        
        try:
            report = self.chain.invoke({
                "query": query,
                "context": context
            })
            
            state.draft_report = report
            state.current_agent = "synthesis"
            
            logger.info(f"Generated report: {len(report)} characters")
            
            self._display_report(report)
            
            console.print(f"\n[bold green]✅ Research report generated! ({len(report)} characters)[/bold green]")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            console.print(f"[red]❌ Report generation failed: {e}[/red]")
            state.error = f"Report generation failed: {e}"
        
        return state
    
    def _retrieve_chunks(self, query: str, top_k: int = 20) -> List[Document]:
        """Retrieve relevant chunks from vector store."""
        logger.debug(f"Retrieving top {top_k} chunks for: {query[:50]}...")
        
        try:
            vector_store = get_vector_store()
            chunks = vector_store.search(query, top_k=top_k)
            return chunks
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _format_context(self, chunks: List[Document]) -> str:
        """Format retrieved chunks with enhanced metadata for LLM."""
        context_parts = []
        sources: Dict[str, int] = {}
        
        for chunk in chunks:
            source_url = chunk.metadata.get("source_url", "Unknown")
            title = chunk.metadata.get("title", "Unknown")
            score = chunk.metadata.get("similarity_score", 0)
            source_type = chunk.metadata.get("source_type", "other")
            pub_date = chunk.metadata.get("published_date", "")
            
            # Track unique sources
            if source_url not in sources:
                sources[source_url] = len(sources) + 1
            
            source_num = sources[source_url]
            
            # Add temporal context if date available
            temporal = ""
            if pub_date:
                temporal = f"\n**Date:** {get_temporal_context(pub_date)}"
            
            context_parts.append(
                f"### Source [{source_num}]: {title}\n"
                f"**Type:** {source_type.upper()}\n"
                f"**URL:** {source_url}"
                f"{temporal}\n"
                f"**Relevance:** {score:.3f}\n\n"
                f"{chunk.page_content}\n\n"
                f"---\n"
            )
        
        return "\n".join(context_parts)
    
    def _display_report(self, report: str) -> None:
        """Display the generated report."""
        preview = report[:1500] + "..." if len(report) > 1500 else report
        
        console.print(Panel(
            Markdown(preview),
            title="📄 Report Preview",
            border_style="green"
        ))
        
        console.print(f"\n[dim]Full report length: {len(report)} characters[/dim]")
    
    def synthesize(self, query: str, top_k: int = 20) -> str:
        """Convenience method to generate a report directly."""
        logger.info(f"Direct synthesis for: {query[:50]}...")
        
        chunks = self._retrieve_chunks(query, top_k)
        
        if not chunks:
            logger.warning("No chunks found for synthesis")
            return "No relevant information found in the knowledge base."
        
        context = self._format_context(chunks)
        
        report = self.chain.invoke({
            "query": query,
            "context": context
        })
        
        logger.info(f"Generated report: {len(report)} characters")
        return report


# Singleton instance
synthesis_agent = SynthesisAgent()
