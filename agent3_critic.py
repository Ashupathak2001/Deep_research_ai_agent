"""
Critic Agent: Evaluates and improves research answers
"""

import json
from typing import Dict, Any, List, Optional
from langchain_cohere import ChatCohere
from pydantic import BaseModel, Field, validator

class CriticEvaluation(BaseModel):
    accuracy_score: int = Field(..., ge=1, le=10)
    completeness_score: int = Field(..., ge=1, le=10) 
    clarity_score: int = Field(..., ge=1, le=10)
    overall_score: int = Field(..., ge=1, le=10)
    strengths: List[str] = Field(..., min_items=1)
    weaknesses: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(..., min_items=1)
    revised_answer: Optional[str] = None

class CriticAgent:
    """Agent that critiques and improves research answers"""
    
    def __init__(self, cohere_api_key: str):
        """Initialize with validated API key"""
        if not cohere_api_key or len(cohere_api_key) < 20:
            raise ValueError("Invalid Cohere API key format")
            
        self.llm = ChatCohere(
            model="command",
            cohere_api_key=cohere_api_key,
            temperature=0.3,  # Lower temperature for more consistent evaluations
            max_tokens=10000
        )
    
    def evaluate_answer(self, question: str, answer: str, 
                        research_findings: str, structured_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of an answer and provide improvement suggestions
        """
        try:
            # Truncate inputs if necessary to fit context window
            research_summary = json.dumps(structured_summary.get("summary_text", {}), indent=2)
            
            # Create evaluation prompt
            evaluation_prompt = f"""
            You are an expert research critic evaluating the quality of a research answer.
            
            ORIGINAL QUESTION: {question}
            
            ANSWER TO EVALUATE:
            {answer}
            
            RESEARCH SUMMARY (FOR REFERENCE):
            {research_summary}
            
            Please evaluate this answer on:
            1. Accuracy (1-10): How factually correct and well-supported is the answer?
            2. Completeness (1-10): How thoroughly does it address all aspects of the question?
            3. Clarity (1-10): How well-organized and easy to understand is the answer?
            4. Overall quality (1-10): Your overall assessment considering all factors
            
            Then identify:
            - 2-4 specific strengths of the answer
            - 2-4 specific weaknesses or areas for improvement
            - 3-5 concrete suggestions for improving the answer
            
            Format your response as a valid JSON object with these keys:
            {{
              "accuracy_score": (number 1-10),
              "completeness_score": (number 1-10),
              "clarity_score": (number 1-10),
              "overall_score": (number 1-10),
              "strengths": ["strength1", "strength2", ...],
              "weaknesses": ["weakness1", "weakness2", ...],
              "improvement_suggestions": ["suggestion1", "suggestion2", ...]
            }}
            
            Ensure your response is valid JSON only, with no other text.
            """
            
            # Get evaluation from LLM
            response = self.llm.invoke(evaluation_prompt)
            
            # Parse the evaluation
            evaluation_data = self._extract_evaluation(response.content)
            
            # If score is below threshold, revise the answer
            if evaluation_data.get("overall_score", 0) < 7:
                revised_answer = self.revise_answer(
                    question, 
                    answer, 
                    evaluation_data.get("improvement_suggestions", []),
                    research_findings,
                    structured_summary
                )
                evaluation_data["revised_answer"] = revised_answer
            
            return evaluation_data
            
        except Exception as e:
            return {
                "accuracy_score": 5,
                "completeness_score": 5,
                "clarity_score": 5,
                "overall_score": 5,
                "strengths": ["Could not properly evaluate"],
                "weaknesses": ["Evaluation failed due to technical error"],
                "improvement_suggestions": ["Please review manually"],
                "error": str(e)
            }
    
    def _extract_evaluation(self, content: str) -> Dict[str, Any]:
        """Extract and validate evaluation JSON from LLM response"""
        try:
            # Try direct parsing first
            evaluation = json.loads(content)
            
            # Validate with Pydantic
            validated = CriticEvaluation(**evaluation)
            return validated.dict()
            
        except (json.JSONDecodeError, Exception) as e:
            # Try to extract JSON with regex
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(0)
                    evaluation = json.loads(json_str)
                    validated = CriticEvaluation(**evaluation)
                    return validated.dict()
                except:
                    pass
            
            # Return basic structure if all extraction fails
            return {
                "accuracy_score": 5,
                "completeness_score": 5,
                "clarity_score": 5,
                "overall_score": 5,
                "strengths": ["Could not extract evaluation"],
                "weaknesses": ["Parser error"],
                "improvement_suggestions": ["Review manually"],
                "error": str(e)
            }
    
    def revise_answer(self, question: str, original_answer: str, 
                     improvement_suggestions: List[str], 
                     research_findings: str,
                     structured_summary: Dict[str, Any]) -> str:
        """
        Revise the answer based on evaluation feedback
        """
        try:
            # Create concise summary of findings for context
            summary_json = json.dumps(structured_summary.get("summary_text", {}), indent=2)
            
            suggestions_text = "\n".join([f"- {suggestion}" for suggestion in improvement_suggestions])
            
            revision_prompt = f"""
            You are an expert research editor. You need to improve this research answer based on specific feedback.
            
            ORIGINAL QUESTION: {question}
            
            ORIGINAL ANSWER:
            {original_answer}
            
            IMPROVEMENT SUGGESTIONS:
            {suggestions_text}
            
            RESEARCH SUMMARY FOR REFERENCE:
            {summary_json}
            
            Please revise the original answer to address all the improvement suggestions.
            Make the answer more accurate, complete, and clear while maintaining its overall structure.
            Use the research summary as a reference for factual information.
            
            Your revised answer should be comprehensive but concise, well-structured, and directly address the original question.
            You may add or remove content as needed to improve quality.
            
            Provide only the revised answer with no other text or explanations.
            """
            
            # Get revised answer from LLM
            response = self.llm.invoke(revision_prompt)
            
            return response.content
            
        except Exception as e:
            return f"{original_answer}\n\n[Note: Automatic revision failed: {str(e)}]"