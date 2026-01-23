# agents/reflection.py
"""
Reflection Agent - Evaluates research report quality and decides if revision is needed.

Responsibilities:
- Analyze the generated report against quality criteria
- Score the report on multiple dimensions
- Provide specific feedback for improvement
- Decide whether to accept or revise the report
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json
import re

from utils.config import get_settings
from utils.logger import get_agent_logger
from graph.state import ResearchState

console = Console()
settings = get_settings()
logger = get_agent_logger("reflection")


# Quality evaluation prompt
REFLECTION_PROMPT = """You are a senior research quality evaluator. Your job is to critically assess research reports and provide actionable feedback.

## Research Query
{query}

## Generated Report
{report}

## Evaluation Criteria

Rate the report on each dimension (1-10 scale) and provide specific feedback:

### 1. Completeness (1-10)
- Does it answer the research query fully?
- Are all major aspects covered?
- Any obvious gaps?

### 2. Accuracy & Citations (1-10)
- Are claims properly cited?
- Is information presented accurately?
- Are sources credible?

### 3. Recency (1-10)
- Does it include recent developments (2024-2026)?
- Are sources up-to-date?
- Is temporal context provided?

### 4. Critical Analysis (1-10)
- Does it distinguish hype from reality?
- Are conflicting viewpoints presented?
- Is evidence quality discussed?

### 5. Practical Value (1-10)
- Are there specific examples?
- Is there a comparison table?
- Can the reader take action on this?

### 6. Structure & Clarity (1-10)
- Is it well-organized?
- Is the writing clear?
- Is the length appropriate?

## Output Format

Respond with a JSON object in this EXACT format:
```json
{{
    "scores": {{
        "completeness": 8,
        "accuracy": 7,
        "recency": 6,
        "critical_analysis": 7,
        "practical_value": 8,
        "structure": 9
    }},
    "overall_score": 7.5,
    "verdict": "REVISE",
    "strengths": [
        "Good structure with clear sections",
        "Includes comparison table"
    ],
    "weaknesses": [
        "Missing recent 2025 developments",
        "Lacks specific company examples"
    ],
    "revision_instructions": "Add more recent examples from 2025. Include specific company implementations like IBM, Google, etc. Strengthen the critical analysis section."
}}
```

IMPORTANT:
- "verdict" must be either "ACCEPT" or "REVISE"
- Use "ACCEPT" if overall_score >= 7.5
- Use "REVISE" if overall_score < 7.5 or if there are critical gaps
- Be specific in strengths, weaknesses, and revision instructions

Evaluate the report now:"""


class ReflectionAgent:
    """
    Evaluates research report quality and provides improvement feedback.
    """
    
    def __init__(self):
        logger.info("Initializing Reflection Agent")
        
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.2,  # Low temp for consistent evaluation
            max_tokens=2048
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a meticulous research quality evaluator. You provide fair, constructive feedback with specific examples. Always respond with valid JSON."),
            ("human", REFLECTION_PROMPT)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        logger.info("Reflection Agent initialized")
    
    def run(self, state: ResearchState) -> ResearchState:
        """Main entry point for LangGraph workflow."""
        logger.info("Reflection Agent starting")
        console.print("\n[bold blue]🔍 Reflection Agent Starting...[/bold blue]")
        
        # Get query and report
        query = state.refined_query if state.refined_query else state.original_query
        report = state.draft_report
        
        if not report:
            logger.error("No report to evaluate")
            state.error = "No report available for evaluation"
            return state
        
        console.print("[blue]Evaluating report quality...[/blue]")
        console.print(f"[dim]Report length: {len(report)} characters[/dim]\n")
        
        try:
            # Get evaluation from LLM
            result = self.chain.invoke({
                "query": query,
                "report": report
            })
            
            # Parse the JSON response
            evaluation = self._parse_evaluation(result)
            
            if evaluation:
                # Update state
                state.quality_score = evaluation["overall_score"]
                state.needs_revision = evaluation["verdict"] == "REVISE"
                state.revision_feedback = evaluation.get("revision_instructions", "")
                state.iteration_count += 1
                state.current_agent = "reflection"
                
                # Display results
                self._display_evaluation(evaluation)
                
                if state.needs_revision and state.iteration_count < state.max_iterations:
                    console.print(f"\n[yellow]📝 Revision needed (iteration {state.iteration_count}/{state.max_iterations})[/yellow]")
                elif state.needs_revision:
                    console.print(f"\n[red]⚠️ Max iterations reached. Accepting current report.[/red]")
                    state.needs_revision = False
                else:
                    console.print("\n[bold green]✅ Report meets quality standards![/bold green]")
                    state.final_report = report
                
            else:
                logger.error("Failed to parse evaluation response")
                state.error = "Failed to parse evaluation"
                
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            console.print(f"[red]❌ Evaluation failed: {e}[/red]")
            state.error = f"Reflection failed: {e}"
        
        return state
    
    def _parse_evaluation(self, result: str) -> dict:
        """Parse the JSON evaluation from LLM response."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None
    
    def _display_evaluation(self, evaluation: dict) -> None:
        """Display the evaluation results."""
        scores = evaluation.get("scores", {})
        overall = evaluation.get("overall_score", 0)
        verdict = evaluation.get("verdict", "UNKNOWN")
        
        # Scores table
        table = Table(title="📊 Quality Evaluation")
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", style="yellow", justify="center")
        table.add_column("Status", justify="center")
        
        for dim, score in scores.items():
            status = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
            table.add_row(dim.replace("_", " ").title(), f"{score}/10", status)
        
        # Add overall score
        table.add_row("", "", "")
        overall_status = "✅" if overall >= 7.5 else "⚠️"
        table.add_row("[bold]OVERALL[/bold]", f"[bold]{overall}/10[/bold]", overall_status)
        
        console.print(table)
        
        # Verdict
        verdict_color = "green" if verdict == "ACCEPT" else "yellow"
        console.print(f"\n[bold {verdict_color}]Verdict: {verdict}[/bold {verdict_color}]")
        
        # Strengths
        strengths = evaluation.get("strengths", [])
        if strengths:
            console.print("\n[green]✓ Strengths:[/green]")
            for s in strengths:
                console.print(f"  • {s}")
        
        # Weaknesses
        weaknesses = evaluation.get("weaknesses", [])
        if weaknesses:
            console.print("\n[yellow]⚠ Weaknesses:[/yellow]")
            for w in weaknesses:
                console.print(f"  • {w}")
        
        # Revision instructions
        if verdict == "REVISE":
            instructions = evaluation.get("revision_instructions", "")
            if instructions:
                console.print(Panel(
                    instructions,
                    title="📝 Revision Instructions",
                    border_style="yellow"
                ))
    
    def evaluate(self, query: str, report: str) -> Tuple[float, bool, str]:
        """
        Convenience method to evaluate a report directly.
        
        Returns:
            Tuple of (score, needs_revision, feedback)
        """
        logger.info("Direct evaluation")
        
        result = self.chain.invoke({
            "query": query,
            "report": report
        })
        
        evaluation = self._parse_evaluation(result)
        
        if evaluation:
            return (
                evaluation.get("overall_score", 0),
                evaluation.get("verdict") == "REVISE",
                evaluation.get("revision_instructions", "")
            )
        
        return (0, True, "Evaluation failed")


# Singleton instance
reflection_agent = ReflectionAgent()
