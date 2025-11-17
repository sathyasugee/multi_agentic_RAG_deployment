"""
RAG Agent - Document-based Question Answering
==============================================
This agent answers questions using retrieved documents from vector database.

WORKFLOW:
1. Receive user query
2. Retrieve relevant documents from FAISS
3. Format context for LLM
4. Generate answer using Groq LLM
5. Return answer with source citations
"""

from typing import Dict, Optional
from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Import from parent directory
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from retrieval import Retriever
from prompts import PromptBuilder


class RAGAgent:
    """
    RAG Agent: Answers questions using vector database retrieval

    Think of this as: A librarian who searches documents and explains them
    """

    def __init__(self, retriever: Optional[Retriever] = None):
        """
        Initialize RAG Agent

        Args:
            retriever: Custom retriever (optional, will create if not provided)
        """
        self.config = config

        # Initialize retriever (loads vector store)
        self.retriever = retriever if retriever else Retriever()
        logger.info("✓ RAG Agent: Retriever initialized")

        # Initialize LLM (Groq API)
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.get('llm.temperature', 0.3),
            max_tokens=self.config.get('llm.max_tokens', 1024)
        )
        logger.info(f"✓ RAG Agent: LLM initialized ({self.config.llm_model})")

        # Get prompt template
        self.prompt = PromptBuilder.get_rag_prompt()

        # Create processing chain (LangChain pattern)
        # Chain: Prompt → LLM → String Output
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("✓ RAG Agent: Processing chain ready")

    def answer(self, query: str, top_k: int = None) -> Dict:
        """
        Main method: Answer a question using RAG

        STEP-BY-STEP PROCESS:
        1. Retrieve relevant documents
        2. Format context
        3. Send to LLM with prompt
        4. Parse response
        5. Return structured result

        Args:
            query: User's question
            top_k: Number of documents to retrieve (default from config)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"📚 RAG Agent processing: '{query}'")

        try:
            # STEP 1: Retrieve relevant documents from vector DB
            retrieval_result = self.retriever.get_context_for_query(
                query=query,
                top_k=top_k
            )

            documents = retrieval_result['documents']
            context = retrieval_result['context']

            if not documents:
                logger.warning("⚠ No relevant documents found")
                return {
                    'answer': "I don't have information about that in the documents.",
                    'sources': [],
                    'num_sources': 0,
                    'agent': 'RAG_AGENT'
                }

            logger.info(f"  → Retrieved {len(documents)} documents")

            # STEP 2: Generate answer using LLM
            # The chain automatically:
            # - Formats prompt with context + query
            # - Sends to Groq API
            # - Parses string response
            answer = self.chain.invoke({
                'context': context,
                'question': query
            })

            logger.info(f"  → Generated answer ({len(answer)} chars)")

            # STEP 3: Extract source information
            sources = self._extract_sources(documents)

            # STEP 4: Return structured result
            return {
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources),
                'agent': 'RAG_AGENT',
                'documents': documents  # Full docs for debugging
            }

        except Exception as e:
            logger.error(f"❌ RAG Agent error: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'agent': 'RAG_AGENT',
                'error': str(e)
            }

    def _extract_sources(self, documents) -> list:
        """
        Extract source citations from retrieved documents

        Returns list of source dictionaries with metadata
        """
        sources = []
        for doc in documents:
            source_info = {
                'content_preview': doc.page_content[:100] + '...',
                'metadata': doc.metadata
            }
            sources.append(source_info)

        return sources

    def get_name(self) -> str:
        """Return agent identifier"""
        return "RAG_AGENT"


# ============================================================
# Testing Interface
# ============================================================

def main():
    """Test RAG Agent"""
    logger.info("🧪 Testing RAG Agent\n")

    # Initialize agent
    agent = RAGAgent()

    # Test queries
    test_queries = [
        "What are the gaming license requirements?",
        "Tell me about parking regulations",
        "What is the capital of France?"  # Should say not in docs
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Query: '{query}'")
        logger.info('=' * 60)

        # Get answer
        result = agent.answer(query)

        # Display result
        logger.info(f"\n📝 Answer:")
        logger.info(f"{result['answer']}\n")

        logger.info(f"📚 Sources: {result['num_sources']}")
        for i, source in enumerate(result['sources'], 1):
            logger.info(f"  {i}. {source['content_preview']}")

    logger.info("\n✅ RAG Agent testing completed!")


if __name__ == "__main__":
    main()