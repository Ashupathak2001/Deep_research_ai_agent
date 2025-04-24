import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pipeline import ResearchPipeline
from typing import Any, Dict, List

# Configuration
MAX_HISTORY_ITEMS = 10
MAX_QUERY_LENGTH = 500
MAX_FEEDBACK_LENGTH = 1000


def validate_api_keys(tavily_key: str, cohere_key: str) -> bool:
    return (tavily_key and len(tavily_key) >= 20 and cohere_key and len(cohere_key) >= 20)

def sanitize_input(text: str, max_length: int) -> str:
    return text.strip()[:max_length] if text else ""

def initialize_session():
    if "research_pipeline" not in st.session_state:
        load_dotenv()
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        cohere_key = os.getenv("COHERE_API_KEY", "")
        if not validate_api_keys(tavily_key, cohere_key):
            st.error("Invalid API key format in .env file")
            st.stop()
        st.session_state.research_pipeline = ResearchPipeline(tavily_api_key=tavily_key, cohere_api_key=cohere_key)
    if "research_history" not in st.session_state:
        st.session_state.research_history = []

def display_error(error: str) -> None:
    st.error(f"⚠️ Error: {error}")
    st.stop()

def main():
    st.set_page_config(page_title="Deep Research AI System", page_icon="🔍", layout="wide")

    try:
        initialize_session()
        st.title("Deep Research AI Agentic System")

        with st.form("research_form"):
            research_query = st.text_area(
                "💬 Enter your research question:",
                placeholder="e.g., What are the latest advancements in quantum computing?",
                height=100,
                max_chars=MAX_QUERY_LENGTH
            )
            submitted = st.form_submit_button("🔍 Start Research")

        if submitted:
            research_query = sanitize_input(research_query, MAX_QUERY_LENGTH)
            if not research_query:
                display_error("Please enter a research question")

            with st.spinner("🔎 Researching... please wait..."):
                try:
                    result = st.session_state.research_pipeline.execute(research_query)
                    st.session_state.research_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": research_query,
                        "result": result
                    })
                    if len(st.session_state.research_history) > MAX_HISTORY_ITEMS:
                        st.session_state.research_history.pop(0)
                except Exception as e:
                    display_error(f"Research failed: {str(e)}")

        if st.session_state.research_history:
            display_results()

    except Exception as e:
        display_error(f"Application error: {str(e)}")

def display_results():
    tab1, tab2 = st.tabs(["📌 Current Result", "📜 History"])
    with tab1:
        latest = st.session_state.research_history[-1]
        if latest["result"].get("errors"):
            st.error("\n".join(latest["result"]["errors"]))
        st.markdown(f"### ❓ Question\n{latest.get('question', '')}")
        st.markdown("### ✅ Final Answer")
        final_answer = latest["result"].get("final_answer", "") or latest["result"].get("answer_draft", "No answer generated.")
        st.markdown(final_answer)
        with st.expander("📊 View Quality Evaluation"):
            display_evaluation(latest["result"])
        with st.expander("📚 View Research Details"):
            display_research_details(latest["result"])
        display_feedback_section(latest)
    with tab2:
        display_history()

def display_evaluation(result: Dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if not evaluation:
        st.info("No evaluation data available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{evaluation.get('accuracy_score', 'N/A')}/10")
    col2.metric("Completeness", f"{evaluation.get('completeness_score', 'N/A')}/10")
    col3.metric("Clarity", f"{evaluation.get('clarity_score', 'N/A')}/10")
    col4.metric("Overall", f"{evaluation.get('overall_score', 'N/A')}/10")

    colL, colR = st.columns(2)
    with colL:
        st.markdown("#### 🌟 Strengths")
        for s in evaluation.get("strengths", []):
            st.markdown(f"✅ {s}")
    with colR:
        st.markdown("#### ⚠️ Weaknesses")
        for w in evaluation.get("weaknesses", []):
            st.markdown(f"🔍 {w}")
    st.markdown("#### 💡 Suggestions")
    for s in evaluation.get("improvement_suggestions", []):
        st.markdown(f"💡 {s}")
    if evaluation.get("revised_answer") and evaluation["revised_answer"] != result.get("final_answer"):
        with st.expander("📝 View Original Draft"):
            st.markdown(result.get("answer_draft", ""))

def display_history() -> None:
    if len(st.session_state.research_history) > 1:
        for i, item in enumerate(reversed(st.session_state.research_history[:-1])):
            with st.expander(f"{item['timestamp'][:10]}: {item['question'][:50]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['result'].get('final_answer', '') or item['result'].get('answer_draft', '')}")
    else:
        st.info("No previous research history.")

def display_research_details(result: Dict[str, Any]) -> None:
    st.markdown("#### 📑 Structured Summary")
    summary = result.get("structured_summary", {})
    st.json(summary.get("summary_text", summary))
    st.markdown("#### 📖 Raw Research Findings")
    st.text(result.get("research_findings", "")[:10000])

def display_feedback_section(item: Dict[str, Any]) -> None:
    st.markdown("### ✍️ Provide Feedback")
    with st.form("feedback_form"):
        feedback = st.text_area(
            "💡 Help us refine the answer:",
            placeholder="Enter your suggestions here...",
            height=100,
            max_chars=MAX_FEEDBACK_LENGTH
        )
        if st.form_submit_button("🔄 Refine Answer"):
            feedback = sanitize_input(feedback, MAX_FEEDBACK_LENGTH)
            if not feedback:
                st.warning("Please enter feedback")
                return
            with st.spinner("Refining answer..."):
                try:
                    refined = st.session_state.research_pipeline.execute(
                        item["question"],
                        feedback=feedback,
                        needs_refinement=True
                    )
                    item["result"] = refined
                    st.rerun()
                except Exception as e:
                    st.error(f"Refinement failed: {str(e)}")

if __name__ == "__main__":
    main()
