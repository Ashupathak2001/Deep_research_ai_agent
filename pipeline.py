"""
Improved Pipeline with Critic agent integration
"""

import time
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ValidationError
from agent1_researcher import ResearcherAgent, ResearchResult
from agent2_drafter import DrafterAgent
from agent3_critic import CriticAgent

class ResearchState(BaseModel):
    question: str
    research_findings: str = ""
    structured_summary: Dict[str, Any] = {}
    answer_draft: str = ""
    structured_components: Dict[str, Any] = {}
    evaluation: Dict[str, Any] = {}
    final_answer: str = ""
    errors: List[str] = []
    feedback: str = ""
    needs_refinement: bool = False
    
    class Config:
        extra = "forbid"  # Prevent extra fields

class ResearchPipeline:
    MAX_EXECUTION_TIME = 300  # 5 minutes
    MAX_HISTORY_SIZE = 20
    
    def __init__(self, tavily_api_key: str, cohere_api_key: str):
        """Initialize with validated configuration"""
        self._validate_api_keys(tavily_api_key, cohere_api_key)
        
        self.researcher = ResearcherAgent(tavily_api_key, cohere_api_key)
        self.drafter = DrafterAgent(cohere_api_key)
        self.critic = CriticAgent(cohere_api_key)
        self.workflow = self._build_graph()
        self.execution_history = []
    
    def _validate_api_keys(self, tavily_key: str, cohere_key: str) -> None:
        """Validate API key formats"""
        if not tavily_key or len(tavily_key) < 20:
            raise ValueError("Invalid Tavily API key")
        if not cohere_key or len(cohere_key) < 20:
            raise ValueError("Invalid Cohere API key")
    
    def _validate_state(self, state: Dict[str, Any]) -> ResearchState:
        """Validate and clean state"""
        try:
            return ResearchState(**state)
        except ValidationError as e:
            raise ValueError(f"Invalid state: {str(e)}")
    
    def _research_task(self, state: ResearchState) -> Dict[str, Any]:
        """Research task with timeout"""
        start_time = time.time()
        
        try:
            result = self.researcher.research(state.question)
            
            if time.time() - start_time > self.MAX_EXECUTION_TIME - 30:
                raise TimeoutError("Research took too long")
                
            return {
                "research_findings": result.research_findings,
                "structured_summary": result.structured_summary,
                "errors": state.errors + result.errors
            }
            
        except Exception as e:
            return {
                "research_findings": "",
                "structured_summary": {},
                "errors": state.errors + [f"Research error: {str(e)}"]
            }
    
    def _draft_answer(self, state: ResearchState) -> Dict[str, Any]:
        """Draft answer with validation"""
        try:
            result = self.drafter.draft_answer(
                state.question,
                state.research_findings,
                state.structured_summary
            )
            
            return {
                "answer_draft": result.get("answer_text",""),
                "structured_components": result.get("structured_components", {}),
                "errors": state.errors + result.get("errors", [])
            }
            
        except Exception as e:
            return {
                "answer_draft": f"Drafting failed: {str(e)}",
                "structured_components": {},
                "errors": state.errors + [str(e)]
            }
    
    def _critique_answer(self, state: ResearchState) -> Dict[str, Any]:
        """Critique and potentially revise the answer"""
        try:
            evaluation = self.critic.evaluate_answer(
                state.question,
                state.answer_draft,
                state.research_findings,
                state.structured_summary
            )
            
            # If critic provided a revised answer, use it as the final answer
            final_answer = evaluation.get("revised_answer", state.answer_draft)
            
            return {
                "evaluation": evaluation,
                "final_answer": final_answer,
                "errors": state.errors
            }
            
        except Exception as e:
            return {
                "evaluation": {
                    "accuracy_score": 5,
                    "completeness_score": 5,
                    "clarity_score": 5,
                    "overall_score": 5,
                    "strengths": ["Evaluation failed"],
                    "weaknesses": ["Technical error occurred"],
                    "improvement_suggestions": []
                },
                "final_answer": state.answer_draft,  # Use draft as fallback
                "errors": state.errors + [f"Critique error: {str(e)}"]
            }
    
    def _refine_answer(self, state: ResearchState) -> Dict[str, Any]:
        """Refine answer with checks"""
        if not state.feedback:
            return {
                "answer_draft": state.answer_draft,
                "final_answer": state.final_answer or state.answer_draft,
                "needs_refinement": False,
                "errors": state.errors + ["No feedback provided for refinement"]
            }
            
        try:
            refined = self.drafter.refine_answer(
                state.final_answer or state.answer_draft,
                state.feedback
            )
            
            # After refinement, have the critic evaluate again
            evaluation = self.critic.evaluate_answer(
                state.question,
                refined,
                state.research_findings,
                state.structured_summary
            )
            
            # If critic score improved and there's a revised version, use it
            if (evaluation.get("overall_score", 0) > 
                state.evaluation.get("overall_score", 0) and
                evaluation.get("revised_answer")):
                final_refined = evaluation.get("revised_answer")
            else:
                final_refined = refined
            
            return {
                "answer_draft": refined,
                "final_answer": final_refined,
                "evaluation": evaluation,
                "needs_refinement": False
            }
            
        except Exception as e:
            return {
                "answer_draft": state.answer_draft,
                "final_answer": state.final_answer or state.answer_draft,
                "needs_refinement": True,
                "errors": state.errors + [f"Refinement failed: {str(e)}"]
            }
    
    def _should_refine(self, state: ResearchState) -> str:
        """Conditional routing with validation"""
        if state.needs_refinement and state.feedback:
            return "refine"
        return "critique"  # Always go to critique after drafting
    
    def _after_critique(self, state: ResearchState) -> str:
        """Decide what to do after critique"""
        if state.needs_refinement and state.feedback:
            return "refine"
        return END
    
    def _build_graph(self) -> StateGraph:
        """Build validated workflow graph"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("research", self._research_task)
        workflow.add_node("draft", self._draft_answer)
        workflow.add_node("critique", self._critique_answer)
        workflow.add_node("refine", self._refine_answer)
        
        # Connect nodes
        workflow.add_edge("research", "draft")
        workflow.add_edge("draft", "critique")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "critique",
            self._after_critique,
            {
                "refine": "refine",
                END: END
            }
        )
        workflow.add_edge("refine", END)
        
        workflow.set_entry_point("research")
        return workflow.compile()
    
    def execute(self, question: str, feedback: str = "", 
               needs_refinement: bool = False) -> Dict[str, Any]:
        """
        Execute pipeline with timeout and state validation
        """
        start_time = time.time()
        
        try:
            initial_state = ResearchState(
                question=question,
                feedback=feedback,
                needs_refinement=needs_refinement
            )
            
            result = self.workflow.invoke(initial_state.dict())
            
            # Track execution history
            self._update_history(result)
            
            return result
            
        except Exception as e:
            return {
                "question": question,
                "answer_draft": f"Pipeline execution failed: {str(e)}",
                "final_answer": f"Pipeline execution failed: {str(e)}",
                "errors": [str(e)],
                "feedback": feedback,
                "needs_refinement": False
            }
    
    def _update_history(self, result: Dict[str, Any]) -> None:
        """Manage execution history with size limits"""
        self.execution_history.append(result)
        if len(self.execution_history) > self.MAX_HISTORY_SIZE:
            self.execution_history.pop(0)