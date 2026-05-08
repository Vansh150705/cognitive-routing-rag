import os
import json
from typing import TypedDict
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.8,
    api_key=os.getenv("GROQ_API_KEY"),
)

MOCK_NEWS_DB = {
    "crypto":     "Bitcoin hits new all-time high amid regulatory ETF approvals; altcoins rally 30%.",
    "bitcoin":    "Bitcoin hits new all-time high amid regulatory ETF approvals; altcoins rally 30%.",
    "ai":         "OpenAI releases GPT-5 — insiders say it clears PhD-level benchmarks across every domain.",
    "openai":     "OpenAI releases GPT-5 — insiders say it clears PhD-level benchmarks across every domain.",
    "elon":       "Elon Musk's xAI surpasses Google Search in monthly active users for the first time.",
    "space":      "SpaceX Starship completes first crewed lunar flyby, six months ahead of NASA schedule.",
    "market":     "S&P 500 breaks 6,000 as tech earnings crush expectations; VIX drops to 10-year low.",
    "stocks":     "S&P 500 breaks 6,000 as tech earnings crush expectations; VIX drops to 10-year low.",
    "fed":        "Fed signals three rate cuts in 2025; dollar weakens, gold surges past $2,800.",
    "rates":      "Fed signals three rate cuts in 2025; dollar weakens, gold surges past $2,800.",
    "privacy":    "EU regulators fine Meta €1.3 billion for cross-border data transfers without consent.",
    "meta":       "EU regulators fine Meta €1.3 billion for cross-border data transfers without consent.",
    "monopoly":   "DOJ opens new antitrust probe into Apple, Google, and Amazon on the same day.",
    "layoffs":    "Google, Microsoft, and Amazon announce combined 40,000 layoffs driven by AI automation.",
    "automation": "Google, Microsoft, and Amazon announce combined 40,000 layoffs driven by AI automation.",
}
DEFAULT_HEADLINE = "Tech stocks surge as AI investment cycle shows no signs of slowing."

@tool
def mock_searxng_search(query: str) -> str:
    """Simulate a SearxNG web search with hardcoded headlines."""
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            return headline
    return DEFAULT_HEADLINE

class PostState(TypedDict):
    bot_id:        str
    persona:       str
    search_query:  str
    search_result: str
    post_json:     dict

def decide_search(state: PostState) -> PostState:
    system = (
        "You are the internal reasoning layer of a social-media bot. "
        "Your persona: " + state["persona"] + "\n\n"
        "Pick ONE topic that interests you and write a short (≤8 words) "
        "web-search query. Reply with ONLY the search query."
    )
    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="What do you want to post about today?")])
    query = response.content.strip().strip('"')
    print(f"  [decide_search] query: {query!r}")
    return {**state, "search_query": query}

def web_search(state: PostState) -> PostState:
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  [web_search] result: {result!r}")
    return {**state, "search_result": result}

def draft_post(state: PostState) -> PostState:
    system = (
        "You are a social-media bot. Your persona:\n" + state["persona"] + "\n\n"
        "Using the context below, write a single highly opinionated post "
        "of at most 280 characters. Stay in character — no disclaimers.\n\n"
        f"Context: {state['search_result']}\n\n"
        "Respond with ONLY a valid JSON object, no markdown, no code fences.\n"
        'Schema: {"bot_id": string, "topic": string, "post_content": string}'
    )
    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="Draft your post now.")])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    post_data = json.loads(raw)
    post_data["bot_id"] = state["bot_id"]
    if len(post_data.get("post_content", "")) > 280:
        post_data["post_content"] = post_data["post_content"][:277] + "..."
    print(f"  [draft_post] {json.dumps(post_data, ensure_ascii=False)}")
    return {**state, "post_json": post_data}

def build_content_graph():
    graph = StateGraph(PostState)
    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search",    web_search)
    graph.add_node("draft_post",    draft_post)
    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)
    return graph.compile()

def run_content_engine(bot_id: str, persona: str) -> dict:
    app = build_content_graph()
    final_state = app.invoke({
        "bot_id": bot_id, "persona": persona,
        "search_query": "", "search_result": "", "post_json": {},
    })
    return final_state["post_json"]

if __name__ == "__main__":
    from phase1_router import BOT_PERSONAS
    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\n{'='*60}\n[bot] {bot_id}")
        result = run_content_engine(bot_id, persona)
        print(f"\n[output JSON]\n{json.dumps(result, indent=2)}")