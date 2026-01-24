# agents/clarification.py
"""
Clarification Agent - Refines user queries through intelligent follow-up questions.

Responsibilities:
- Analyze the user's initial query for ambiguity or missing context
- Generate targeted clarification questions
- Incorporate user responses to create a refined, focused query
- Support Human-in-the-Loop (HITL) interaction
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import json
import re

from utils.config import get_settings
from utils.logger import get_agent_logger
from graph.state import ResearchState

console = Console()
settings = get_settings()
logger = get_agent_logger("clarification")


# Prompt for analyzing query and generating questions
CLARIFICATION_PROMPT = """You are a research assistant helping to clarify and refine research queries.

## User's Original Query
{query}

## Your Task
Analyze this query and determine if clarification is needed. Consider:
1. **Scope**: Is the query too broad or too narrow?
2. **Time frame**: Does the user want recent info, historical, or both?
3. **Perspective**: Academic, industry, practical applications, or all?
4. **Specifics**: Are there ambiguous terms or multiple interpretations?
5. **Depth**: Quick overview or deep dive?

## Response Format
Respond with JSON in this EXACT format:
```json
{{
    "needs_clarification": true,
    "analysis": "Brief analysis of what's unclear or could be refined",
    "questions": [
        "Question 1 about scope or specifics?",
        "Question 2 about time frame or perspective?",
        "Question 3 about depth or focus area?"
    ],
    "suggested_refined_query": "A more specific version if clarification isn't needed"
}}
```

If the query is already clear and specific, set "needs_clarification": false.
Maximum 3 questions, each focused on one aspect.

Analyze the query now:"""


# Prompt for refining query based on responses
REFINEMENT_PROMPT = """Based on the original query and the user's clarification responses, create a refined research query.

## Original Query
{original_query}

## Clarification Q&A
{qa_pairs}

## Instructions
Create a refined, specific research query that:
1. Incorporates all the user's preferences and specifications
2. Is focused and actionable for research
3. Includes any time frame, scope, or perspective constraints mentioned

Respond with ONLY the refined query, nothing else."""


class ClarificationAgent:
    """
    Refines user queries through clarification questions.
    Supports Human-in-the-Loop interaction.
    """
    
    def __init__(self):
        logger.info("Initializing Clarification Agent")
        
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.3
        )
        
        # Chain for analyzing query
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful research assistant that asks clarifying questions. Always respond with valid JSON."),
            ("human", CLARIFICATION_PROMPT)
        ])
        self.analysis_chain = self.analysis_prompt | self.llm | StrOutputParser()
        
        # Chain for refining query
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research query optimizer. Create clear, specific research queries."),
            ("human", REFINEMENT_PROMPT)
        ])
        self.refinement_chain = self.refinement_prompt | self.llm | StrOutputParser()
        
        logger.info("Clarification Agent initialized")
    
    def run(self, state: ResearchState) -> ResearchState:
        """
        Main entry point for LangGraph workflow.
        Analyzes query and optionally asks clarification questions.
        """
        logger.info("Clarification Agent starting")
        console.print("\n[bold blue]💬 Clarification Agent Starting...[/bold blue]")
        
        query = state.original_query
        
        if not query:
            logger.error("No query provided")
            state.error = "No query provided for clarification"
            return state
        
        console.print(f"[cyan]Original Query:[/cyan] {query}\n")
        
        # Analyze the query
        console.print("[blue]Analyzing query for clarity...[/blue]")
        analysis = self._analyze_query(query)
        
        if not analysis:
            logger.warning("Failed to analyze query, proceeding without clarification")
            state.refined_query = query
            state.clarification_complete = True
            return state
        
        needs_clarification = analysis.get("needs_clarification", False)
        questions = analysis.get("questions", [])
        
        if not needs_clarification or not questions:
            # Query is already clear
            console.print("[green]✓ Query is clear and specific![/green]")
            refined = analysis.get("suggested_refined_query", query)
            state.refined_query = refined
            state.clarification_complete = True
            
            if refined != query:
                console.print(f"[cyan]Refined Query:[/cyan] {refined}")
            
            logger.info("No clarification needed")
            return state
        
        # Display analysis
        console.print(Panel(
            analysis.get("analysis", ""),
            title="📋 Query Analysis",
            border_style="yellow"
        ))
        
        # Store questions in state
        state.clarification_questions = questions
        
        # Ask questions (HITL)
        console.print("\n[bold yellow]Please answer the following questions:[/bold yellow]\n")
        
        responses = []
        for i, question in enumerate(questions, 1):
            console.print(f"[cyan]Q{i}:[/cyan] {question}")
            response = Prompt.ask(f"   [dim]Your answer[/dim]")
            responses.append(response)
            console.print()
        
        state.user_responses = responses
        
        # Refine the query based on responses
        console.print("[blue]Refining query based on your responses...[/blue]")
        refined_query = self._refine_query(query, questions, responses)
        
        state.refined_query = refined_query
        state.clarification_complete = True
        state.current_agent = "clarification"
        
        console.print(Panel(
            refined_query,
            title="✨ Refined Research Query",
            border_style="green"
        ))
        
        logger.info(f"Query refined: {refined_query[:50]}...")
        
        return state
    
    def _analyze_query(self, query: str) -> Optional[dict]:
        """Analyze query and determine if clarification is needed."""
        try:
            result = self.analysis_chain.invoke({"query": query})
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
            return None
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return None
    
    def _refine_query(self, original: str, questions: List[str], responses: List[str]) -> str:
        """Refine the query based on Q&A responses."""
        try:
            # Format Q&A pairs
            qa_pairs = "\n".join([
                f"Q: {q}\nA: {r}" 
                for q, r in zip(questions, responses)
            ])
            
            result = self.refinement_chain.invoke({
                "original_query": original,
                "qa_pairs": qa_pairs
            })
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return original
    
    def clarify(self, query: str) -> Tuple[str, List[str], List[str]]:
        """
        Convenience method for direct clarification.
        
        Returns:
            Tuple of (refined_query, questions, responses)
        """
        logger.info(f"Direct clarification for: {query[:50]}...")
        
        # Analyze
        analysis = self._analyze_query(query)
        
        if not analysis or not analysis.get("needs_clarification"):
            return (analysis.get("suggested_refined_query", query), [], [])
        
        questions = analysis.get("questions", [])
        
        # Get responses (interactive)
        responses = []
        for question in questions:
            response = Prompt.ask(f"{question}")
            responses.append(response)
        
        # Refine
        refined = self._refine_query(query, questions, responses)
        
        return (refined, questions, responses)
    
    def analyze_query_api(self, query: str) -> dict:
        """
        API-friendly method: Analyze query and return clarification info.
        Does NOT require interactive input.
        
        Returns:
            {
                "needs_clarification": bool,
                "analysis": str,
                "questions": List[str],
                "suggested_refined_query": str
            }
        """
        logger.info(f"API clarification analysis for: {query[:50]}...")
        
        analysis = self._analyze_query(query)
        
        if not analysis:
            return {
                "needs_clarification": False,
                "analysis": "Unable to analyze query",
                "questions": [],
                "suggested_refined_query": query
            }
        
        return {
            "needs_clarification": analysis.get("needs_clarification", False),
            "analysis": analysis.get("analysis", ""),
            "questions": analysis.get("questions", []),
            "suggested_refined_query": analysis.get("suggested_refined_query", query)
        }
    
    def refine_query_api(self, original_query: str, questions: List[str], responses: List[str]) -> str:
        """
        API-friendly method: Refine query based on user's answers.
        
        Args:
            original_query: The original user query
            questions: List of clarification questions that were asked
            responses: List of user's answers to those questions
            
        Returns:
            Refined query string
        """
        logger.info(f"API query refinement for: {original_query[:50]}...")
        
        if not questions or not responses:
            return original_query
        
        return self._refine_query(original_query, questions, responses)


# Singleton instance
clarification_agent = ClarificationAgent()

