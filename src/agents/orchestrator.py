"""
Orchestrator Agent - Multi-Agent Coordinator
============================================
This is the BRAIN of the system. It decides which agent(s) to use.

WORKFLOW:
1. Analyze incoming query
2. Detect keywords/patterns
3. Route to appropriate agent(s)
4. Synthesize results if multiple agents used
5. Return final answer
"""

from typing import Dict, Optional
from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config
from prompts import PromptBuilder
from agents.rag_agent import RAGAgent
from agents.web_search_agent import WebSearchAgent


class Orchestrator:
    """
    Orchestrator: The intelligent router that coordinates multiple agents

    Think of this as: A project manager who assigns tasks to specialists
    """

    def __init__(self):
        """Initialize Orchestrator with all agents"""
        self.config = config

        # Initialize both agents
        logger.info("🧠 Initializing Orchestrator...")

        self.rag_agent = RAGAgent()
        self.web_search_agent = WebSearchAgent()

        # Initialize LLM for synthesis (when using both agents)
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.get('llm.temperature', 0.3),
            max_tokens=self.config.get('llm.max_tokens', 1024)
        )

        # Get synthesis prompt
        self.synthesis_prompt = PromptBuilder.get_synthesis_prompt()
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()

        # Get routing keywords from config
        self.rag_keywords = self.config.get('agents.routing.rag_keywords', [])
        self.web_keywords = self.config.get('agents.routing.web_keywords', [])

        logger.info("✓ Orchestrator ready with 2 agents")

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine which agent(s) to use

        ROUTING LOGIC:
        1. Count RAG keywords (code, regulation, section, etc.)
        2. Count WEB keywords (recent, update, news, etc.)
        3. Decide routing based on keyword scores

        Args:
            query: User's question

        Returns:
            Dictionary with routing decision and reasoning
        """
        query_lower = query.lower()

        # Count keyword matches
        rag_score = sum(1 for kw in self.rag_keywords if kw in query_lower)
        web_score = sum(1 for kw in self.web_keywords if kw in query_lower)

        # Routing decision logic
        if rag_score > 0 and web_score > 0:
            decision = "BOTH"
            reasoning = f"Query contains both RAG keywords ({rag_score}) and WEB keywords ({web_score})"
        elif web_score > 0:
            decision = "WEB_SEARCH_AGENT"
            reasoning = f"Query contains WEB keywords ({web_score}), needs current information"
        elif rag_score > 0:
            decision = "RAG_AGENT"
            reasoning = f"Query contains RAG keywords ({rag_score}), search documents"
        else:
            # Default to RAG if unsure
            decision = "RAG_AGENT"
            reasoning = "No specific keywords detected, defaulting to document search"

        logger.info(f"🎯 Routing decision: {decision}")
        logger.info(f"   Reasoning: {reasoning}")

        return {
            'decision': decision,
            'reasoning': reasoning,
            'rag_score': rag_score,
            'web_score': web_score
        }

    def execute(self, query: str) -> Dict:
        """
        Main method: Route query and get answer

        COMPLETE ORCHESTRATION:
        1. Analyze query → determine routing
        2. Execute appropriate agent(s)
        3. Synthesize if both agents used
        4. Return final result

        Args:
            query: User's question

        Returns:
            Dictionary with final answer and metadata
        """
        logger.info(f"\n🧠 Orchestrator processing: '{query}'")

        try:
            # STEP 1: Analyze and route
            routing = self.analyze_query(query)
            decision = routing['decision']

            # STEP 2: Execute based on decision
            if decision == "RAG_AGENT":
                # Use RAG agent only
                result = self.rag_agent.answer(query)
                result['routing'] = routing
                return result

            elif decision == "WEB_SEARCH_AGENT":
                # Use Web Search agent only
                result = self.web_search_agent.search_and_answer(query)
                result['routing'] = routing
                return result

            elif decision == "BOTH":
                # Use both agents and synthesize results
                logger.info("  → Executing BOTH agents...")

                # Get answer from RAG agent
                rag_result = self.rag_agent.answer(query)
                logger.info(f"  → RAG Agent: {len(rag_result['answer'])} chars")

                # Get answer from Web Search agent
                web_result = self.web_search_agent.search_and_answer(query)
                logger.info(f"  → Web Search Agent: {len(web_result['answer'])} chars")

                # Synthesize both answers
                synthesized_answer = self._synthesize_answers(
                    query=query,
                    rag_answer=rag_result['answer'],
                    web_answer=web_result['answer']
                )

                logger.info(f"  → Synthesized: {len(synthesized_answer)} chars")

                # Combine sources from both agents
                all_sources = rag_result.get('sources', []) + web_result.get('sources', [])

                return {
                    'answer': synthesized_answer,
                    'sources': all_sources,
                    'num_sources': len(all_sources),
                    'agent': 'BOTH (RAG + WEB)',
                    'routing': routing,
                    'rag_result': rag_result,
                    'web_result': web_result
                }

        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'agent': 'ORCHESTRATOR',
                'error': str(e)
            }

    def _synthesize_answers(self, query: str, rag_answer: str, web_answer: str) -> str:
        """
        Synthesize answers from both agents using LLM

        The LLM reads both answers and creates a unified, coherent response

        Args:
            query: Original question
            rag_answer: Answer from RAG agent
            web_answer: Answer from Web Search agent

        Returns:
            Synthesized final answer
        """
        try:
            synthesized = self.synthesis_chain.invoke({
                'rag_answer': rag_answer,
                'web_answer': web_answer,
                'question': query
            })
            return synthesized

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fallback: concatenate answers
            return f"Based on documents:\n{rag_answer}\n\nBased on web search:\n{web_answer}"


# ============================================================
# Testing Interface
# ============================================================

def main():
    """Test Orchestrator with all routing scenarios"""
    logger.info("🧪 Testing Orchestrator - Multi-Agent System\n")

    # Initialize orchestrator
    orchestrator = Orchestrator()

    # Test queries covering all routing scenarios
    test_queries = [
        # Scenario 1: RAG only
        "What does Section 5:23 say about gaming licenses?",

        # Scenario 2: Web Search only
        "What are the latest AI developments this week?",

        # Scenario 3: BOTH agents
        "Compare NJ gaming regulations with recent federal updates"
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"TEST {i}: {query}")
        logger.info('=' * 70)

        # Execute orchestrator
        result = orchestrator.execute(query)

        # Display results
        logger.info(f"\n🎯 Routing Decision: {result['routing']['decision']}")
        logger.info(f"   {result['routing']['reasoning']}")

        logger.info(f"\n📝 Final Answer:")
        logger.info(f"{result['answer'][:300]}...")

        logger.info(f"\n📚 Total Sources: {result['num_sources']}")

        if result.get('agent') == 'BOTH (RAG + WEB)':
            logger.info("\n🔄 Multi-Agent Synthesis:")
            logger.info(f"  - RAG sources: {result['rag_result']['num_sources']}")
            logger.info(f"  - Web sources: {result['web_result']['num_sources']}")

    logger.info("\n✅ Orchestrator testing completed!")
    logger.info("🎉 Multi-Agent System fully operational!")


if __name__ == "__main__":
    main()