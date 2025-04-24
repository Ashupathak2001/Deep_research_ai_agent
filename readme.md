#  Dual-Agent AI Research System

An intelligent, agent-based research assistant built with LangChain, Tavily, Cohere, and Streamlit. 
This system simulates a team of agents — a Researcher, a Drafter, and a Critic — to deliver high-quality answers backed by real-time web research.

---

##  Features

🔍 **Agent 1 - Researcher**
- Uses Tavily API for deep web search
- Summarizes results using Cohere LLM
- Outputs both raw and structured research data

✍️ **Agent 2 - Drafter**
- Analyzes structured summary + research findings
- Generates a professional, well-structured answer
- Validates with schema (key points, evidence, sources)

🧪 **Agent 3 - Critic**
- Evaluates answer based on accuracy, clarity, completeness
- Provides suggestions & scores (1–10 scale)
- Automatically refines the answer if needed

🌐 **Streamlit UI**
- Clean & styled interface
- Input questions, get smart answers
- Feedback form + real-time answer refinement
- History tracking + quality metrics display

---

## 📦 Project Structure

```bash
├── agent1_researcher.py    # Web search + summarization logic
├── agent2_drafter.py       # Answer generation logic
├── agent3_critic.py        # Critique and auto-refinement
├── pipeline.py             # Workflow management (LangGraph)
├── app.py                  # Streamlit frontend UI
├── .env                    # API keys (Tavily & Cohere)
└── README.md               # You're reading it
```

⚙️ Installation
1. Clone this repo

git clone https://github.com/yourusername/ai-research-agents.git
cd ai-research-agents

2. Create .env file
env

TAVILY_API_KEY=your_tavily_api_key
COHERE_API_KEY=your_cohere_api_key

3. Install dependencies

pip install -r requirements.txt
Run the app

4. streamlit run app.py

🛠 Requirements

Python 3.9+

Streamlit

LangChain

Tavily SDK

Cohere SDK

pydantic

dotenv

👉 You can generate requirements.txt via:

pip freeze > requirements.txt

How It Works
User inputs a research question

Agent1 runs multi-query search and summarizes findings

Agent2 drafts an answer using the findings

Agent3 critiques the answer, scores it, and improves it if needed

Final answer + evaluation are shown in UI

User can provide feedback and request refinement


✨ Future Ideas
Add PDF/CSV export

Add citations tooltips and footnotes

Multilingual support

Dark mode toggle

Feedback-based model fine-tuning

🙌 Credits
Built with using:

LangChain

Tavily

Cohere

Streamlit

📬 Contact
For queries, drop me a message at [ashupathak22@gmail.com] or connect on LinkedIn