"""
Web Search Agent - Internet Research
=====================================
This agent searches the internet for current/external information.

WORKFLOW:
1. Receive user query
2. Search internet using DuckDuckGo
3. Parse and format search results
4. Synthesize answer using LLM
5. Return answer with source URLs
"""

from typing import Dict, List
from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Import from parent directory
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from prompts import PromptBuilder

# Try new package first, fallback to old
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS

        logger.warning("Using deprecated duckduckgo_search. Install ddgs instead: pip install ddgs")
    except ImportError:
        logger.error("No DuckDuckGo package found. Install: pip install ddgs")
        DDGS = None


class WebSearchAgent:
    """
    Web Search Agent: Answers questions using internet search

    Think of this as: A researcher who Googles and summarizes findings
    """

    def __init__(self):
        """Initialize Web Search Agent"""
        self.config = config

        # Initialize LLM (Groq API)
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.get('llm.temperature', 0.3),
            max_tokens=self.config.get('llm.max_tokens', 1024)
        )
        logger.info(f"✓ Web Search Agent: LLM initialized ({self.config.llm_model})")

        # Get prompt template
        self.prompt = PromptBuilder.get_web_search_prompt()

        # Create processing chain
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Search configuration
        self.max_results = self.config.get('agents.max_search_results', 3)
        logger.info(f"✓ Web Search Agent: Ready (max {self.max_results} results)")

    def search_internet(self, query: str, max_results: int = None) -> List[Dict]:
        """
        Search the internet using DuckDuckGo

        STEP-BY-STEP:
        1. Send query to DuckDuckGo API
        2. Get top results (titles, snippets, URLs)
        3. Parse and structure results

        Args:
            query: Search query
            max_results: Number of results to retrieve

        Returns:
            List of search result dictionaries
        """
        if DDGS is None:
            logger.error("DuckDuckGo search not available. Install: pip install ddgs")
            return []

        max_res = max_results or self.max_results

        logger.info(f"🌐 Searching internet: '{query}'")

        try:
            # Use DuckDuckGo API with error handling
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_res))

            # Structure results
            structured_results = []
            for i, result in enumerate(results, 1):
                structured_results.append({
                    'rank': i,
                    'title': result.get('title', 'No title'),
                    'snippet': result.get('body', 'No description'),
                    'url': result.get('href', 'No URL')
                })

            logger.info(f"  → Found {len(structured_results)} results")
            return structured_results

        except Exception as e:
            logger.error(f"❌ Web search error: {e}")
            logger.info("💡 If search fails repeatedly, DuckDuckGo may be rate-limiting")
            return []

    def format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for LLM consumption

        Converts list of results into readable text block
        """
        if not results:
            return "No search results found."

        formatted = []
        for result in results:
            formatted.append(f"""
Result {result['rank']}: {result['title']}
URL: {result['url']}
Content: {result['snippet']}
""")

        return "\n---\n".join(formatted)

    def search_and_answer(self, query: str, max_results: int = None, simulate: bool = False) -> Dict:
        """
        Main method: Search internet and generate answer

        COMPLETE WORKFLOW:
        1. Search internet (DuckDuckGo)
        2. Format results
        3. Send to LLM for synthesis
        4. Return answer with sources

        Args:
            query: User's question
            max_results: Number of search results
            simulate: If True, use simulated results (for demo/testing)

        Returns:
            Dictionary with answer and source URLs
        """
        logger.info(f"🌐 Web Search Agent processing: '{query}'")

        try:
            # STEP 1: Search internet (or simulate)
            if simulate or DDGS is None:
                logger.info("  → Using simulated web results (demo mode)")
                search_results = self._get_simulated_results(query)
            else:
                search_results = self.search_internet(query, max_results)

            if not search_results:
                logger.warning("⚠ No search results found")
                return {
                    'answer': "I couldn't find relevant information on the internet.",
                    'sources': [],
                    'num_sources': 0,
                    'agent': 'WEB_SEARCH_AGENT'
                }

            # STEP 2: Format results for LLM
            formatted_results = self.format_search_results(search_results)

            # STEP 3: Generate synthesized answer using LLM
            # The LLM reads all search results and creates a coherent answer
            answer = self.chain.invoke({
                'search_results': formatted_results,
                'question': query
            })

            logger.info(f"  → Generated answer ({len(answer)} chars)")

            # STEP 4: Extract source URLs
            sources = [
                {'title': r['title'], 'url': r['url']}
                for r in search_results
            ]

            return {
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources),
                'agent': 'WEB_SEARCH_AGENT',
                'raw_results': search_results  # For debugging
            }

        except Exception as e:
            logger.error(f"❌ Web Search Agent error: {e}")
            return {
                'answer': f"Error searching internet: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'agent': 'WEB_SEARCH_AGENT',
                'error': str(e)
            }

    def _get_simulated_results(self, query: str) -> List[Dict]:
        """
        Generate simulated search results for demo purposes
        Used when real search is unavailable
        """
        return [
            {
                'rank': 1,
                'title': f'Recent Article About {query}',
                'snippet': f'This simulated result discusses {query} with recent updates and information.',
                'url': 'https://example.com/article1'
            },
            {
                'rank': 2,
                'title': f'Official Source on {query}',
                'snippet': f'Government or official documentation regarding {query} and related topics.',
                'url': 'https://example.com/article2'
            },
            {
                'rank': 3,
                'title': f'News Update: {query}',
                'snippet': f'Latest news and developments about {query} from reliable sources.',
                'url': 'https://example.com/article3'
            }
        ]

    def get_name(self) -> str:
        """Return agent identifier"""
        return "WEB_SEARCH_AGENT"


# ============================================================
# Testing Interface
# ============================================================

def main():
    """Test Web Search Agent"""
    logger.info("🧪 Testing Web Search Agent\n")

    # Initialize agent
    agent = WebSearchAgent()

    # Test queries
    test_queries = [
        "What are recent AI developments in 2024?",
        "Current weather in New Jersey",
        "Latest news about gaming regulations"
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Query: '{query}'")
        logger.info('=' * 60)

        # Get answer
        result = agent.search_and_answer(query)

        # Display result
        logger.info(f"\n📝 Answer:")
        logger.info(f"{result['answer']}\n")

        logger.info(f"🔗 Sources: {result['num_sources']}")
        for i, source in enumerate(result['sources'], 1):
            logger.info(f"  {i}. {source['title']}")
            logger.info(f"     {source['url']}")

    logger.info("\n✅ Web Search Agent testing completed!")


if __name__ == "__main__":
    main()