"""
Agent 1: Researcher - Simplified Version with Pipeline Compatibility
Handles web crawling and information gathering using Tavily API
and summarizing results using Cohere API
"""
import re
import os
import json
import time
from typing import Dict, Any, List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from langchain_cohere import ChatCohere
from pydantic import BaseModel

class ResearchResult(BaseModel):
    research_findings: str
    structured_summary: Dict[str, Any]
    errors: List[str] = []

class ResearcherAgent:
    MAX_QUERY_LENGTH = 500
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    def __init__(self, tavily_api_key: str, cohere_api_key: str):
        """
        Initialize the Researcher Agent with validated API keys
        """
        if not tavily_api_key or len(tavily_api_key) < 20:
            raise ValueError("Invalid Tavily API key format")
        if not cohere_api_key or len(cohere_api_key) < 20:
            raise ValueError("Invalid Cohere API key format")
            
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        self.llm = ChatCohere(
            model="command",
            cohere_api_key=cohere_api_key,
            temperature=0.2,
            max_tokens=10000
        )
        
        # Create a Tool object to maintain compatibility with pipeline
        self.search_tool = Tool(
            name="web_search",
            description="Search the web for specific information about a topic",
            func=TavilySearchResults(max_results=7, search_depth="advanced").run
        )
        
        # Store tools as a list for compatibility
        self.tools = [self.search_tool]
    
    def _safe_search_execution(self, query: str) -> List[Dict[str, Any]]:
        """Wrapper with retry logic for search operations"""
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.search_tool.run(query)
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(self.RETRY_DELAY * (attempt + 1))
        return []
    
    def _validate_question(self, question: str) -> str:
        """Clean and validate research question"""
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        question = question.strip()
        if len(question) > self.MAX_QUERY_LENGTH:
            question = question[:self.MAX_QUERY_LENGTH]
        return question
    
    def _extract_json_safely(self, content: str) -> Dict[Any, Any]:
        """
        Extract JSON from LLM output with multiple fallback strategies
        """
        # Try direct parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Find JSON-like structure with regex
        import re
        json_pattern = r'\{(?:[^{}]|(?R))*\}'
        matches = re.findall(json_pattern, content)
        
        for potential_json in matches:
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                continue
        
        # Attempt to fix common JSON issues
        # 1. Fix single quotes
        fixed_content = content.replace("'", '"')
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass
        
        # 2. Fix unquoted keys
        try:
            import re
            # Find keys that aren't quoted
            unquoted_key_pattern = r'(\s*?)(\w+)(\s*?):(\s*?)'
            fixed_content = re.sub(unquoted_key_pattern, r'\1"\2"\3:\4', content)
            return json.loads(fixed_content)
        except (json.JSONDecodeError, Exception):
            pass
        
        # 3. Extract key-value pairs with heuristics if all else fails
        structured_data = {}
        lines = content.split('\n')
        current_key = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for key-value pattern
            kv_match = re.match(r'"?([^":]+)"?\s*:\s*(.+)', line)
            if kv_match:
                key, value = kv_match.groups()
                key = key.strip('"\'').strip()
                value = value.strip('"\'').strip().rstrip(',')
                
                # Try to parse value as list or nested structure
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except:
                        value = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
                
                structured_data[key] = value
                current_key = key
            elif current_key and line.startswith('-'):
                # Handle list items
                if isinstance(structured_data.get(current_key), list):
                    structured_data[current_key].append(line[1:].strip())
                else:
                    structured_data[current_key] = [line[1:].strip()]
        
        return structured_data if structured_data else {"error": "Could not parse JSON"}

    def _generate_search_queries(self, question: str) -> List[str]:
        """Use LLM to generate multiple search queries for the question"""
        try:
            prompt = f"""
            I need to research this question thoroughly: "{question}"
            
            Please generate 3-5 different search queries that would help gather comprehensive information about this topic.
            Each query should focus on a different aspect of the question.
            
            Format your response as a JSON array of strings ONLY. For example:
            ["first search query", "second search query", "third search query"]
            """
            
            response = self.llm.invoke(prompt)
            
            # Try to extract queries using robust JSON parsing
            try:
                content = response.content
                
                # Try direct JSON parsing first
                try:
                    queries = json.loads(content)
                    if isinstance(queries, list) and len(queries) > 0:
                        return queries[:5]  # Limit to 5 queries
                except json.JSONDecodeError:
                    pass
                
                # Extract array-like structure with regex
                import re
                array_pattern = r'\[(.*?)\]'
                array_match = re.search(array_pattern, content, re.DOTALL)
                
                if array_match:
                    array_content = array_match.group(1)
                    # Split by commas not inside quotes
                    items = re.findall(r'"([^"]*)"', array_content)
                    if items:
                        return items[:5]
                
                # Last resort: extract quoted strings
                quoted_strings = re.findall(r'"([^"]*)"', content)
                if quoted_strings:
                    return quoted_strings[:5]
                    
                # If all else fails, generate simple queries from the question
                return self._generate_fallback_queries(question)
                    
            except Exception:
                return self._generate_fallback_queries(question)
                    
        except Exception as e:
            print(f"Error generating search queries: {str(e)}")
            return self._generate_fallback_queries(question)
            
    def _generate_fallback_queries(self, question: str) -> List[str]:
        """Generate simple search queries from the question"""
        # Remove question words and common stopwords
        clean_q = re.sub(r'^(what|how|why|when|where|who|is|are|do|does|can|could|would|should)\s+', '', question.lower())
        
        # Generate simple variations
        queries = [
            question,  # Original question
            clean_q,   # Question without question words
            f"latest research {clean_q}",  # Add "latest research"
            f"overview of {clean_q}"  # Add "overview of"
        ]
    
        return [q for q in queries if q.strip()]
    
    def research(self, question: str) -> ResearchResult:
        """Perform research without using ReAct agent"""
        try:
            # Validate the question first
            validated_question = self._validate_question(question)
            
            # Generate multiple search queries
            search_queries = self._generate_search_queries(validated_question)
            
            # Execute searches
            all_results = []
            errors = []
            
            for query in search_queries:
                try:
                    results = self._safe_search_execution(query)
                    if results:
                        all_results.append({
                            "query": query,
                            "results": results
                        })
                except Exception as search_error:
                    errors.append(f"Search failed for query '{query}': {str(search_error)}")
            
            if not all_results:
                return ResearchResult(
                    research_findings="No search results found",
                    structured_summary={
                        "summary_text": {
                            "key_facts": ["No information found"],
                            "statistics": [],
                            "perspectives": [],
                            "consensus": [],
                            "disagreements": [],
                            "sources": []
                        },
                        "processed_successfully": False
                    },
                    errors=errors
                )
            
            # Format the findings
            findings = self._format_findings(validated_question, all_results)
            
            # Summarize the findings
            summary = self.summarize_findings(findings, validated_question)
            
            return ResearchResult(
                research_findings=findings,
                structured_summary=summary,
                errors=errors
            )
                
        except Exception as e:
            return ResearchResult(
                research_findings=f"Research on '{question}' failed",
                structured_summary={
                    "summary_text": {
                        "key_facts": ["Research failed due to technical issues"],
                        "statistics": [],
                        "perspectives": [],
                        "consensus": [],
                        "disagreements": [],
                        "sources": []
                    },
                    "processed_successfully": False
                },
                errors=[f"Research failed: {str(e)}"]
            )
    
    def _format_findings(self, question: str, all_results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable format with size limit"""
        MAX_FINDINGS_SIZE = 8000  # Leave some buffer for other content
        
        findings = f"RESEARCH FINDINGS FOR: {question}\n\n"
        remaining_space = MAX_FINDINGS_SIZE - len(findings)
        
        for result_set in all_results:
            if remaining_space <= 0:
                break
                
            query_section = f"SEARCH QUERY: {result_set['query']}\n" + "=" * 50 + "\n\n"
            if len(query_section) > remaining_space:
                findings += query_section[:remaining_space]
                break
                
            findings += query_section
            remaining_space -= len(query_section)
            
            for idx, result in enumerate(result_set['results'], 1):
                result_text = (
                    f"Result {idx}:\n"
                    f"Title: {result.get('title', 'No title')}\n"
                    f"URL: {result.get('url', 'No URL')}\n"
                    f"Content: {result.get('content', 'No content')}\n\n"
                )
                
                if len(result_text) > remaining_space:
                    findings += result_text[:remaining_space]
                    remaining_space = 0
                    break
                    
                findings += result_text
                remaining_space -= len(result_text)
            
            if remaining_space > 0:
                findings += "-" * 50 + "\n\n"
                remaining_space -= len("-" * 50 + "\n\n")
        
        return findings
    
    def summarize_findings(self, findings: str, question: str) -> Dict[str, Any]:
        """Generate structured summary with robust parsing"""
        # if not findings or len(findings) > 4000:
        #     return {
        #         "summary_text": {
        #             "key_facts": ["Invalid or oversized findings"],
        #             "statistics": [],
        #             "perspectives": [],
        #             "consensus": [],
        #             "disagreements": [],
        #             "sources": []
        #         },
        #         "processed_successfully": False
        #     }
        
        try:
            summary_prompt = f"""
            Below are research findings about this question: "{question}"
            
            RESEARCH FINDINGS:
            {findings[:4000]}  # Truncate if too long
            
            Create a structured summary of these findings with the following components:
            1. Key facts and information
            2. Important statistics or numerical data
            3. Different perspectives or viewpoints
            4. Areas of consensus
            5. Areas of disagreement or uncertainty
            6. Sources referenced (with URLs when available)
            
            Format your response as a structured JSON with these keys:
            {{
            "key_facts": [list of facts],
            "statistics": [list of statistics],
            "perspectives": [list of different viewpoints],
            "consensus": [areas of agreement],
            "disagreements": [areas of disagreement],
            "sources": [list of sources]
            }}
            
            Ensure your response is valid JSON without any other text before or after.
            """
            
            response = self.llm.invoke(summary_prompt)
            content = response.content
            
            # Try multiple JSON extraction strategies
            json_data = self._extract_json_safely(content)
            
            # Ensure all required keys exist
            required_keys = ["key_facts", "statistics", "perspectives", 
                            "consensus", "disagreements", "sources"]
            for key in required_keys:
                if key not in json_data:
                    json_data[key] = []
            
            return {
                "summary_text": json_data,
                "processed_successfully": True
            }
                
        except Exception as e:
            return {
                "summary_text": {
                    "key_facts": [f"Summary generation failed: {str(e)}"],
                    "statistics": [],
                    "perspectives": [],
                    "consensus": [],
                    "disagreements": [],
                    "sources": []
                },
                "processed_successfully": False
            }
    
    def _create_fallback_summary(self, content: str) -> Dict[str, Any]:
        """Create a basic summary from unstructured text"""
        lines = content.split('\n')
        
        summary = {
            "key_facts": [],
            "statistics": [],
            "perspectives": [],
            "consensus": [],
            "disagreements": [],
            "sources": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            lower_line = line.lower()
            
            if "key fact" in lower_line or "key information" in lower_line:
                current_section = "key_facts"
            elif "statistic" in lower_line or "numerical" in lower_line:
                current_section = "statistics"
            elif "perspective" in lower_line or "viewpoint" in lower_line:
                current_section = "perspectives"
            elif "consensus" in lower_line or "agreement" in lower_line:
                current_section = "consensus"
            elif "disagreement" in lower_line or "uncertainty" in lower_line:
                current_section = "disagreements"
            elif "source" in lower_line or "url" in lower_line or "http" in lower_line:
                current_section = "sources"
            elif current_section and line.startswith(("-", "•", "*", "1.", "2.", "3.")):
                # Extract the content after the list marker
                item = line.split(" ", 1)
                if len(item) > 1:
                    summary[current_section].append(item[1])
                else:
                    summary[current_section].append(line)
        
        # If we couldn't extract structured info, add the whole content as a key fact
        if not any(summary.values()):
            summary["key_facts"] = [content[:500]]  # Truncate if too long
        
        return {
            "summary_text": summary,
            "processed_successfully": False  # Mark as not fully processed
        }