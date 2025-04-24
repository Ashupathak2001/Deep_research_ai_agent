"""
Improved Drafter Agent with validation and safety checks
"""

import json
from typing import Dict, Any, List, Optional
from langchain_cohere import ChatCohere
from pydantic import BaseModel, Field, validator, ValidationError
import re

class StructuredAnswer(BaseModel):
    main_answer: str = Field(..., min_length=10)
    key_points: List[str] = Field(..., min_items=1)
    supporting_evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    
    @validator('sources')
    def validate_sources(cls, v):
        """Validate URLs in sources"""
        url_pattern = re.compile(r'https?://\S+')
        return [url for url in v if url_pattern.match(url)]

class DrafterAgent:
    MAX_INPUT_LENGTH = 8000
    MAX_OUTPUT_LENGTH = 4000
    
    def __init__(self, cohere_api_key: str):
        """Initialize with validated API key"""
        if not cohere_api_key or len(cohere_api_key) < 20:
            raise ValueError("Invalid Cohere API key format")
            
        self.llm = ChatCohere(
            model="command",
            cohere_api_key=cohere_api_key,
            temperature=0.4,
            max_tokens=10000
        )
    
    def _validate_inputs(self, question: str, research_findings: str) -> None:
        """Validate all inputs before processing"""
        if not question or len(question) > 500:
            raise ValueError("Invalid question length")
        if not research_findings:
            raise ValueError("Empty research findings")
        if len(research_findings) > self.MAX_INPUT_LENGTH:
            research_findings = research_findings[:self.MAX_INPUT_LENGTH]
    
    def draft_answer(self, question: str, research_findings: str, 
                   structured_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer with comprehensive validation
        """
        try:
            self._validate_inputs(question, research_findings)
            
            # Truncate inputs if necessary
            research_findings = research_findings[:self.MAX_INPUT_LENGTH]
            
            draft_prompt = f"""
            You are an expert answer drafter for a research system. Based on the research findings provided, create a clear, comprehensive answer to the original question.

            QUESTION: {question}

            RESEARCH FINDINGS:
            {research_findings}

            STRUCTURED SUMMARY:
            {json.dumps(structured_summary, indent=2)}

            Your answer should:
            1. Directly address the original question
            2. Be well-structured with clear sections
            3. Include facts and data from the research
            4. Acknowledge any limitations or uncertainties
            5. Not exceed {self.MAX_OUTPUT_LENGTH} characters
            6. Be written in a professional tone

            Format your answer in markdown with appropriate headers and emphasis.
            """
            
            response = self.llm.invoke(draft_prompt)
            draft = response.content[:self.MAX_OUTPUT_LENGTH]
            
            # Generate and validate structured components
            structured = self._generate_structured_components(draft)
            
            return {
                "answer_text": draft,
                "structured_components": structured,
                "errors": []
            }
            
        except Exception as e:
            return {
                "answer_text": f"Error generating answer: {str(e)}",
                "structured_components": {},
                "errors": [str(e)]
            }
    
    def _generate_structured_components(self, draft: str) -> Dict[str, Any]:
        """Generate and validate structured components"""
        try:
            prompt = f"""
            [Strict JSON generation prompt...]
            """
            
            response = self.llm.invoke(prompt)
            
            # Validate and parse JSON
            components = json.loads(response.content)
            validated = StructuredAnswer(**components)
            return validated.dict()
            
        except (json.JSONDecodeError, ValidationError) as e:
            return {
                "error": f"Invalid structured components: {str(e)}"
            }
    
    def refine_answer(self, draft_answer: str, feedback: str) -> str:
        """Refine answer with validation"""
        if not draft_answer or len(draft_answer) > self.MAX_OUTPUT_LENGTH:
            return "Invalid draft answer"
        if not feedback or len(feedback) > 1000:
            return "Invalid feedback"
            
        try:
            refine_prompt = f"""
            [Improved refinement prompt...]
            """
            
            response = self.llm.invoke(refine_prompt)
            return response.content[:self.MAX_OUTPUT_LENGTH]
            
        except Exception as e:
            return f"Refinement failed: {str(e)}"