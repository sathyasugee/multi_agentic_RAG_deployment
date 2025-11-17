"""
Prompt Templates for Multi-Agent RAG System
===========================================

PROMPT ENGINEERING TECHNIQUES USED:
1. Chat Templates (System + User messages)
2. Few-Shot Learning (examples)
3. Chain-of-Thought (reasoning steps)
4. Context Injection (retrieved documents)
5. Tool Use Instructions (multi-agent)
6. Structured Output (JSON/formatting)

Industry Best Practice: Use LangChain's ChatPromptTemplate
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================
# TECHNIQUE 1: SYSTEM MESSAGE PATTERN
# ============================================================
# System messages define the AI's role, behavior, and constraints
# This sets the foundation for ALL interactions

RAG_SYSTEM_MESSAGE = """You are an expert assistant specialized in New Jersey Administrative Code.

Your responsibilities:
- Answer questions ONLY using the provided context documents
- If the answer is not in the context, say "I don't have that information in the documents"
- Cite specific sections/codes when possible
- Be precise and factual
- Use clear, professional language

CRITICAL RULES:
1. Never make up information
2. Always ground answers in the provided context
3. If uncertain, express that clearly
"""

# ============================================================
# TECHNIQUE 2: CONTEXT INJECTION PATTERN
# ============================================================
# RAG-specific: How to inject retrieved documents into prompts
# Uses {context} placeholder that gets filled dynamically

RAG_USER_TEMPLATE = """Context Documents:
{context}

---

User Question: {question}

Instructions:
1. Read the context carefully
2. Find relevant information to answer the question
3. Provide a clear, concise answer
4. Cite the source if possible

Answer:"""

# ============================================================
# TECHNIQUE 3: CHAT TEMPLATE (Industry Standard)
# ============================================================
# Combines system + user messages into a structured template
# This is what modern LLMs (GPT-4, Claude, Llama) expect

RAG_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),
    ("user", RAG_USER_TEMPLATE)
])

# ============================================================
# TECHNIQUE 4: FEW-SHOT LEARNING
# ============================================================
# Teach the model by showing examples of good behavior
# Especially useful for formatting and reasoning

FEW_SHOT_RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),

    # Example 1: Good citation
    ("user", """Context: Section 5:23-2.1 states that all gaming licenses expire annually.
Question: When do gaming licenses expire?"""),
    ("assistant", "According to Section 5:23-2.1, all gaming licenses expire annually."),

    # Example 2: Admitting uncertainty
    ("user", """Context: The document discusses parking regulations in Newark.
Question: What are the speed limits in Jersey City?"""),
    ("assistant",
     "I don't have information about Jersey City speed limits in the provided context. The document only covers parking regulations in Newark."),

    # Actual query
    ("user", RAG_USER_TEMPLATE)
])

# ============================================================
# TECHNIQUE 5: CHAIN-OF-THOUGHT (CoT) PROMPTING
# ============================================================
# Forces the model to show reasoning steps
# Improves accuracy by making thinking explicit

COT_RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),
    ("user", """Context Documents:
{context}

---

User Question: {question}

Think step-by-step:
1. What is the user really asking?
2. Which parts of the context are relevant?
3. What is the direct answer?
4. Are there any important caveats or additional details?

Provide your reasoning, then give the final answer.

Response:""")
])

# ============================================================
# TECHNIQUE 6: MULTI-AGENT ORCHESTRATOR PROMPT
# ============================================================
# Teaches the orchestrator when to use which agent
# This is the "router" for your multi-agent system

ORCHESTRATOR_SYSTEM_MESSAGE = """You are an intelligent routing agent that decides which specialized agent should handle a query.

Available Agents:
1. RAG_AGENT - Answers questions using New Jersey Administrative Code documents
2. WEB_SEARCH_AGENT - Searches the internet for recent/external information

Routing Rules:
- Use RAG_AGENT when the question is about NJ regulations, codes, rules, or legal requirements
- Use WEB_SEARCH_AGENT when the question asks about recent updates, news, current events, or comparisons
- Use BOTH agents if the question requires combining local docs with external context

Respond with: RAG_AGENT, WEB_SEARCH_AGENT, or BOTH
"""

ORCHESTRATOR_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ORCHESTRATOR_SYSTEM_MESSAGE),
    ("user", "Query: {query}\n\nRoute this query:"),
])

# ============================================================
# TECHNIQUE 7: WEB SEARCH AGENT PROMPT
# ============================================================
# Specialized prompt for internet search tasks
# Includes query refinement instructions

WEB_SEARCH_SYSTEM_MESSAGE = """You are a web research assistant that searches the internet for information.

Your process:
1. Analyze the user's question
2. Formulate effective search queries
3. Synthesize results from multiple sources
4. Provide accurate, up-to-date information with source citations

Always cite sources with URLs when possible.
"""

WEB_SEARCH_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", WEB_SEARCH_SYSTEM_MESSAGE),
    ("user", """Search Results:
{search_results}

---

Original Question: {question}

Synthesize the search results to answer the question. Include source URLs.

Answer:""")
])

# ============================================================
# TECHNIQUE 8: QUERY REFINEMENT PROMPT
# ============================================================
# Pre-processes user queries to improve retrieval
# Expands abbreviations, adds context, fixes typos

QUERY_REFINEMENT_TEMPLATE = PromptTemplate.from_template(
    """Refine the following user query to make it better for document retrieval:
    
    Original Query: {query}
    
    Refinement guidelines:
    - Expand abbreviations (NJ → New Jersey)
    - Add relevant context keywords
    - Make it more specific
    - Fix obvious typos
    - Keep it concise (under 20 words)
    
    Refined Query:"""
)

# ============================================================
# TECHNIQUE 9: RESPONSE SYNTHESIS (Multi-Agent Fusion)
# ============================================================
# Combines outputs from multiple agents
# Used when both RAG and Web Search are involved

SYNTHESIS_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a synthesis agent that combines information from multiple sources.

Your task:
1. Integrate information from RAG documents and web search
2. Resolve any conflicts by noting discrepancies
3. Provide a coherent, unified answer
4. Cite sources clearly (Documents vs. Web)
"""),
    ("user", """RAG Agent Answer:
{rag_answer}

Web Search Answer:
{web_answer}

---

Original Question: {question}

Synthesize a comprehensive answer:""")
])

# ============================================================
# TECHNIQUE 10: STRUCTURED OUTPUT WITH JSON
# ============================================================
# Forces specific output format for parsing
# Critical for multi-agent coordination

JSON_OUTPUT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You must respond ONLY with valid JSON. No other text.

Format:
{
  "answer": "The actual answer",
  "confidence": 0.0-1.0,
  "sources": ["source1", "source2"],
  "requires_web_search": true/false
}"""),
    ("user", "{query}")
])


# ============================================================
# UTILITY: PROMPT BUILDER CLASS
# ============================================================
# Easy access to all templates with customization

class PromptBuilder:
    """Centralized prompt management"""

    @staticmethod
    def get_rag_prompt(use_cot: bool = False, use_few_shot: bool = False) -> ChatPromptTemplate:
        """Get RAG prompt with optional enhancements"""
        if use_few_shot:
            return FEW_SHOT_RAG_TEMPLATE
        elif use_cot:
            return COT_RAG_TEMPLATE
        else:
            return RAG_CHAT_TEMPLATE

    @staticmethod
    def get_orchestrator_prompt() -> ChatPromptTemplate:
        """Get multi-agent routing prompt"""
        return ORCHESTRATOR_TEMPLATE

    @staticmethod
    def get_web_search_prompt() -> ChatPromptTemplate:
        """Get web search agent prompt"""
        return WEB_SEARCH_TEMPLATE

    @staticmethod
    def get_synthesis_prompt() -> ChatPromptTemplate:
        """Get multi-agent fusion prompt"""
        return SYNTHESIS_TEMPLATE

    @staticmethod
    def get_query_refinement_prompt() -> PromptTemplate:
        """Get query refinement prompt"""
        return QUERY_REFINEMENT_TEMPLATE


# ============================================================
# TESTING & EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("🎯 Prompt Templates Test\n")

    # Test 1: Basic RAG prompt
    print("1️⃣ Basic RAG Prompt:")
    rag_prompt = PromptBuilder.get_rag_prompt()
    messages = rag_prompt.format_messages(
        context="Section 5:23 covers gaming regulations.",
        question="What does Section 5:23 cover?"
    )
    for msg in messages:
        print(f"  {msg.type.upper()}: {msg.content[:100]}...")

    print("\n" + "=" * 60 + "\n")

    # Test 2: Orchestrator routing
    print("2️⃣ Orchestrator Routing:")
    orch_prompt = PromptBuilder.get_orchestrator_prompt()
    messages = orch_prompt.format_messages(
        query="What are the latest updates to NJ gaming laws?"
    )
    print(f"  {messages[1].content}")

    print("\n" + "=" * 60 + "\n")

    # Test 3: Few-shot template
    print("3️⃣ Few-Shot Template:")
    few_shot = PromptBuilder.get_rag_prompt(use_few_shot=True)
    print(f"  Message count: {len(few_shot.messages)}")
    print(f"  Includes {(len(few_shot.messages) - 2) // 2} examples")

    print("\n✅ All prompt templates loaded successfully!")