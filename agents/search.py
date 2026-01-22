# agents/search.py
"""
Web Search Agent using Tavily API.

Responsibilities:
- Generate diverse search queries from user's research question
- Execute PARALLEL web searches using Tavily
- Ensure SOURCE DIVERSITY (academic, news, industry, etc.)
- Return structured source information with rich metadata
"""

from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import json
import re
from datetime import datetime

from utils.config import get_settings
from utils.logger import get_agent_logger
from graph.state import Source, ResearchState

console = Console()
settings = get_settings()
logger = get_agent_logger("search")

# Domain to source type mapping
DOMAIN_TYPES = {
    # Academic
    "arxiv.org": "academic",
    "nature.com": "academic",
    "science.org": "academic",
    "sciencedirect.com": "academic",
    "springer.com": "academic",
    "ieee.org": "academic",
    "acm.org": "academic",
    "ncbi.nlm.nih.gov": "academic",
    "pubmed.gov": "academic",
    "researchgate.net": "academic",
    
    # News
    "techcrunch.com": "news",
    "wired.com": "news",
    "theverge.com": "news",
    "reuters.com": "news",
    "bbc.com": "news",
    "cnn.com": "news",
    "nytimes.com": "news",
    "wsj.com": "news",
    "arstechnica.com": "news",
    "technologyreview.com": "news",
    
    # Industry
    "ibm.com": "industry",
    "google.com": "industry",
    "microsoft.com": "industry",
    "amazon.com": "industry",
    "nvidia.com": "industry",
    "intel.com": "industry",
    
    # Wiki
    "wikipedia.org": "wiki",
    
    # Blogs
    "medium.com": "blog",
    "substack.com": "blog",
    "dev.to": "blog",
}


def classify_source_type(url: str) -> str:
    """Classify a URL into a source type category."""
    url_lower = url.lower()
    for domain, source_type in DOMAIN_TYPES.items():
        if domain in url_lower:
            return source_type
    return "other"


def extract_publication(url: str) -> Optional[str]:
    """Extract publication name from URL."""
    url_lower = url.lower()
    
    publications = {
        "nature.com": "Nature",
        "science.org": "Science",
        "arxiv.org": "arXiv",
        "techcrunch.com": "TechCrunch",
        "wired.com": "Wired",
        "theverge.com": "The Verge",
        "reuters.com": "Reuters",
        "bbc.com": "BBC",
        "nytimes.com": "New York Times",
        "wikipedia.org": "Wikipedia",
        "medium.com": "Medium",
    }
    
    for domain, pub in publications.items():
        if domain in url_lower:
            return pub
    return None


class WebSearchAgent:
    """
    Searches the web for relevant sources using Tavily.
    Generates diverse queries and ensures source type diversity.
    """
    
    def __init__(self):
        logger.info("Initializing Web Search Agent")
        
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)
        
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.7
        )
        
        # Updated prompt for diverse source coverage
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant that generates diverse search queries.
Given a research question, generate {num_queries} different search queries that will find sources from DIFFERENT perspectives:

1. Academic/Research: "site:arxiv.org OR site:nature.com {topic} research paper"
2. Recent News: "{topic} news 2024 2025 announcement"
3. Industry/Commercial: "{topic} company startup commercial application"
4. Technical Deep-Dive: "{topic} how it works technical explanation"
5. Critical Analysis: "{topic} challenges limitations criticism"

Return ONLY a JSON array of query strings, no other text.
Example: ["query 1", "query 2", "query 3"]"""),
            ("human", "Research question: {question}")
        ])
        
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
        self.max_workers = 5
        
        logger.info(f"Web Search Agent initialized with model={settings.llm_model}")
    
    def run(self, state: ResearchState) -> ResearchState:
        """Main entry point for LangGraph workflow."""
        logger.info("Web Search Agent starting")
        console.print("\n[bold blue]🔍 Web Search Agent Starting...[/bold blue]")
        
        query = state.refined_query if state.refined_query else state.original_query
        
        if not query:
            logger.error("No query provided for search")
            state.error = "No query provided for search"
            return state
        
        logger.info(f"Processing query: {query}")
        console.print(f"[cyan]Query:[/cyan] {query}\n")
        
        # Generate diverse search queries
        console.print("[blue]Generating diverse search queries...[/blue]")
        search_queries = self._generate_queries(query, settings.num_search_queries)
        state.search_queries = search_queries
        
        logger.info(f"Generated {len(search_queries)} search queries")
        console.print(f"[green]✓ Generated {len(search_queries)} queries[/green]")
        for i, q in enumerate(search_queries, 1):
            console.print(f"  {i}. {q}")
        
        # Execute searches in parallel
        console.print(f"\n[blue]Searching web in parallel ({self.max_workers} threads)...[/blue]")
        all_results = self._search_parallel(search_queries)
        
        # Process results with rich metadata
        all_sources: List[Source] = []
        seen_urls = set()
        source_type_counts: Dict[str, int] = {}
        
        for results in all_results:
            for result in results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    
                    # Classify and enrich source
                    source_type = classify_source_type(url)
                    publication = extract_publication(url)
                    
                    source = Source(
                        url=url,
                        title=result.get("title", "Unknown"),
                        snippet=result.get("content", "")[:500],
                        approved=False,
                        source_type=source_type,
                        publication=publication,
                        published_date=result.get("published_date")
                    )
                    all_sources.append(source)
                    
                    # Track diversity
                    source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        
        # Update state
        state.sources = all_sources
        state.source_diversity = source_type_counts
        state.current_agent = "search"
        
        # Display results with diversity info
        self._display_sources(all_sources, source_type_counts)
        
        logger.info(f"Search complete. Found {len(all_sources)} unique sources")
        console.print(f"\n[bold green]✅ Found {len(all_sources)} unique sources[/bold green]")
        
        return state
    
    def _search_parallel(self, queries: List[str]) -> List[List[dict]]:
        """Execute multiple searches in parallel."""
        logger.debug(f"Starting parallel search for {len(queries)} queries")
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=len(queries))
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_query = {
                    executor.submit(self._search, query): query 
                    for query in queries
                }
                
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.append(results)
                        logger.debug(f"Search completed for: {query[:50]}...")
                        progress.update(task, description=f"✓ {query[:35]}...")
                    except Exception as e:
                        logger.warning(f"Search error for '{query[:30]}...': {e}")
                        all_results.append([])
                    
                    progress.advance(task)
        
        return all_results
    
    def _generate_queries(self, question: str, num_queries: int = 5) -> List[str]:
        """Generate diverse search queries."""
        logger.debug(f"Generating {num_queries} queries for: {question[:50]}...")
        
        try:
            result = self.query_chain.invoke({
                "question": question,
                "num_queries": num_queries
            })
            
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            queries = json.loads(result)
            
            if isinstance(queries, list):
                logger.debug(f"Successfully generated {len(queries)} queries")
                return queries[:num_queries]
            else:
                return [question]
                
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return [question]
    
    def _search(self, query: str, max_results: int = 5) -> List[dict]:
        """Execute a single Tavily search."""
        response = self.tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False
        )
        return response.get("results", [])
    
    def _display_sources(self, sources: List[Source], diversity: Dict[str, int]) -> None:
        """Display sources with diversity breakdown."""
        if not sources:
            return
        
        # Diversity summary
        console.print("\n[bold]📊 Source Diversity:[/bold]")
        for stype, count in sorted(diversity.items(), key=lambda x: -x[1]):
            emoji = {"academic": "📚", "news": "📰", "industry": "🏢", "wiki": "📖", "blog": "✍️"}.get(stype, "🔗")
            console.print(f"  {emoji} {stype.capitalize()}: {count}")
        
        # Source table
        table = Table(title=f"📚 Discovered Sources ({len(sources)})")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Title", style="white", max_width=45)
        table.add_column("Publication", style="blue", max_width=15)
        
        for i, source in enumerate(sources[:15], 1):  # Show top 15
            title = source.title[:42] + "..." if len(source.title) > 45 else source.title
            pub = source.publication or "-"
            table.add_row(str(i), source.source_type, title, pub)
        
        if len(sources) > 15:
            table.add_row("...", "...", f"({len(sources) - 15} more)", "...")
        
        console.print(table)
    
    def search_single(self, query: str, max_results: int = 10) -> List[Source]:
        """Convenience method for single query search."""
        logger.info(f"Single search: {query}")
        results = self._search(query, max_results)
        sources = []
        
        for result in results:
            url = result.get("url", "")
            source = Source(
                url=url,
                title=result.get("title", "Unknown"),
                snippet=result.get("content", "")[:500],
                source_type=classify_source_type(url),
                publication=extract_publication(url)
            )
            sources.append(source)
        
        return sources


# Singleton instance
web_search_agent = WebSearchAgent()
