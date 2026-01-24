# ui/app.py
"""
Streamlit Frontend for Multi-Agent Research System.
Run with: streamlit run ui/app.py
"""

import streamlit as st
import requests
import time
from typing import Optional

# API Configuration
API_BASE = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="centered"
)

# Custom CSS for clean look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .question-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #6366f1;
    }
    .analysis-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "stage" not in st.session_state:
        st.session_state.stage = "input"  # input, clarifying, researching, complete
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "clarification" not in st.session_state:
        st.session_state.clarification = None
    if "task_id" not in st.session_state:
        st.session_state.task_id = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None


def reset_state():
    """Reset to initial state."""
    st.session_state.stage = "input"
    st.session_state.query = ""
    st.session_state.clarification = None
    st.session_state.task_id = None
    st.session_state.result = None
    st.session_state.error = None


def analyze_query(query: str) -> Optional[dict]:
    """Call /clarify endpoint to analyze query."""
    try:
        response = requests.post(
            f"{API_BASE}/clarify",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.session_state.error = f"Failed to analyze query: {e}"
        return None


def submit_clarification(session_id: str, responses: list) -> Optional[str]:
    """Submit clarification responses and get refined query."""
    try:
        response = requests.post(
            f"{API_BASE}/clarify/respond",
            json={"session_id": session_id, "responses": responses},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("refined_query")
    except Exception as e:
        st.session_state.error = f"Failed to refine query: {e}"
        return None


def start_research(query: str) -> Optional[str]:
    """Start research task and return task_id."""
    try:
        response = requests.post(
            f"{API_BASE}/research",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("task_id")
    except Exception as e:
        st.session_state.error = f"Failed to start research: {e}"
        return None


def check_research_status(task_id: str) -> dict:
    """Check research task status."""
    try:
        response = requests.get(f"{API_BASE}/research/{task_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Initialize
init_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Research Agent</h1>
    <p style="color: #64748b;">Multi-Agent Deep Research System</p>
</div>
""", unsafe_allow_html=True)

# Error display
if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Try Again"):
        reset_state()
        st.rerun()

# Stage: Input
elif st.session_state.stage == "input":
    st.markdown("### What would you like to research?")
    
    query = st.text_input(
        "Research Topic",
        placeholder="Enter your research topic...",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Research", type="primary", disabled=not query.strip()):
            st.session_state.query = query.strip()
            st.session_state.stage = "analyzing"
            st.rerun()
    
    # Example queries
    st.markdown("**Try these:**")
    examples = [
        "AI agents in healthcare 2024",
        "Latest advances in RAG systems",
        "Multi-agent research systems"
    ]
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.query = example
                st.session_state.stage = "analyzing"
                st.rerun()

# Stage: Analyzing query for clarification
elif st.session_state.stage == "analyzing":
    with st.spinner("Analyzing your query..."):
        result = analyze_query(st.session_state.query)
        
        if result:
            st.session_state.clarification = result
            if result.get("needs_clarification") and result.get("questions"):
                st.session_state.stage = "clarifying"
            else:
                # No clarification needed, start research
                refined = result.get("suggested_refined_query", st.session_state.query)
                task_id = start_research(refined)
                if task_id:
                    st.session_state.task_id = task_id
                    st.session_state.stage = "researching"
        st.rerun()

# Stage: Clarification questions
elif st.session_state.stage == "clarifying":
    clarification = st.session_state.clarification
    
    st.markdown("### 💬 Let me understand better")
    
    # Analysis box
    st.markdown(f"""
    <div class="analysis-box">
        {clarification.get('analysis', '')}
    </div>
    """, unsafe_allow_html=True)
    
    # Questions
    questions = clarification.get("questions", [])
    answers = []
    
    for i, question in enumerate(questions):
        st.markdown(f"**Q{i+1}:** {question}")
        answer = st.text_input(
            f"Answer {i+1}",
            key=f"q_{i}",
            placeholder="Your answer...",
            label_visibility="collapsed"
        )
        answers.append(answer)
    
    # Suggested query
    suggested = clarification.get("suggested_refined_query", "")
    if suggested:
        st.markdown(f"**Suggested query:** _{suggested}_")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        all_answered = all(a.strip() for a in answers)
        if st.button("Continue with Answers", type="primary", disabled=not all_answered):
            refined = submit_clarification(clarification["session_id"], answers)
            if refined:
                task_id = start_research(refined)
                if task_id:
                    st.session_state.task_id = task_id
                    st.session_state.stage = "researching"
                    st.rerun()
    
    with col2:
        if st.button("Skip (use suggested)"):
            task_id = start_research(suggested)
            if task_id:
                st.session_state.task_id = task_id
                st.session_state.stage = "researching"
                st.rerun()
    
    with col3:
        if st.button("Cancel"):
            reset_state()
            st.rerun()

# Stage: Researching
elif st.session_state.stage == "researching":
    st.markdown("### 🔬 Researching...")
    st.markdown(f"*\"{st.session_state.query}\"*")
    
    progress = st.progress(0)
    status_text = st.empty()
    
    # Stages display
    stages = ["🔍 Searching web", "📊 Analyzing sources", "📝 Synthesizing report", "✅ Quality review"]
    stage_display = st.empty()
    
    # Poll for results
    task_id = st.session_state.task_id
    progress_value = 0
    
    while True:
        result = check_research_status(task_id)
        status = result.get("status", "unknown")
        
        if status == "completed":
            st.session_state.result = result
            st.session_state.stage = "complete"
            st.rerun()
            break
        elif status == "failed":
            st.session_state.error = result.get("error", "Research failed")
            st.rerun()
            break
        else:
            # Update progress
            progress_value = min(progress_value + 5, 95)
            progress.progress(progress_value)
            
            # Show current stage
            current_stage_idx = min(progress_value // 25, 3)
            stage_display.markdown(" → ".join([
                f"**{s}**" if i == current_stage_idx else s 
                for i, s in enumerate(stages)
            ]))
            
            status_text.text(f"Status: {status}")
            time.sleep(3)

# Stage: Complete
elif st.session_state.stage == "complete":
    result = st.session_state.result
    
    st.success("### ✅ Research Complete")
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sources Analyzed", result.get("sources_found", 0))
    with col2:
        score = result.get("quality_score")
        if score:
            st.metric("Quality Score", f"{score}/10")
    
    # Report
    st.markdown("---")
    report = result.get("report", "No report generated")
    st.markdown(report)
    
    # Download button
    st.download_button(
        "📥 Download Report",
        report,
        file_name="research_report.md",
        mime="text/markdown"
    )
    
    # New research
    if st.button("🔄 New Research"):
        reset_state()
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Powered by Multi-Agent Research System • Clarify → Search → Analyze → Synthesize → Reflect"
    "</p>",
    unsafe_allow_html=True
)
