"""
Retrieval System for Multi-Agent RAG
=====================================
Handles semantic search and context preparation for LLM
"""

from typing import List, Dict, Optional
from loguru import logger

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import config
from data_ingestion import DataIngestion


class Retriever:
    """Semantic search and context retrieval"""

    def __init__(self, vectorstore: FAISS = None):
        self.config = config
        self.vectorstore = vectorstore

        # Load vectorstore if not provided
        if self.vectorstore is None:
            logger.info("📂 Loading vector store...")
            ingestion = DataIngestion()
            self.vectorstore = ingestion.load_vectorstore()
            logger.info("✓ Vector store loaded")

    def search(
            self,
            query: str,
            top_k: int = None,
            score_threshold: float = None,
            filter_metadata: Dict = None
    ) -> List[Document]:
        """
        Semantic search in vector database

        Args:
            query: Search query
            top_k: Number of results (default from config)
            score_threshold: Minimum similarity score
            filter_metadata: Filter by metadata (e.g., {'source': 'nj_admin_code'})

        Returns:
            List of relevant documents
        """
        k = top_k or self.config.top_k
        threshold = score_threshold or self.config.get('retrieval.score_threshold', 0.0)

        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter_metadata
        )

        # Filter by score threshold
        filtered_docs = [
            doc for doc, score in results
            if score >= threshold
        ]

        logger.info(f"🔍 Retrieved {len(filtered_docs)}/{k} documents for: '{query[:50]}...'")
        return filtered_docs

    def search_with_scores(
            self,
            query: str,
            top_k: int = None
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self.config.top_k
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(f"🔍 Retrieved {len(results)} documents with scores")
        return results

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string for LLM

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Format each document with metadata
            doc_text = f"Document {i}:\n"
            doc_text += f"Content: {doc.page_content}\n"

            # Add relevant metadata
            if 'source' in doc.metadata:
                doc_text += f"Source: {doc.metadata['source']}\n"
            if 'row_id' in doc.metadata:
                doc_text += f"Row ID: {doc.metadata['row_id']}\n"

            context_parts.append(doc_text)

        return "\n---\n".join(context_parts)

    def get_context_for_query(
            self,
            query: str,
            top_k: int = None
    ) -> Dict[str, any]:
        """
        Complete retrieval pipeline for a query

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with documents and formatted context
        """
        # Retrieve documents
        documents = self.search(query, top_k=top_k)

        # Format context
        context = self.format_context(documents)

        return {
            'query': query,
            'documents': documents,
            'context': context,
            'num_results': len(documents)
        }

    def get_relevant_metadata(self, documents: List[Document]) -> List[Dict]:
        """Extract and return metadata from documents"""
        return [doc.metadata for doc in documents]


# ============================================================
# Testing Interface
# ============================================================

def main():
    """Test retrieval system"""

    # Initialize retriever
    logger.info("🚀 Testing Retrieval System\n")
    retriever = Retriever()

    # Test queries
    test_queries = [
        "gaming license requirements",
        "regulations for parking",
        "safety standards"
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Query: '{query}'")
        logger.info('=' * 60)

        # Get context
        result = retriever.get_context_for_query(query, top_k=3)

        # Display results
        logger.info(f"\nRetrieved {result['num_results']} documents:")
        for i, doc in enumerate(result['documents'], 1):
            logger.info(f"\n  📄 Document {i}:")
            logger.info(f"     Text: {doc.page_content[:150]}...")
            logger.info(f"     Score: {doc.metadata.get('score', 'N/A')}")

        # Show formatted context (first 500 chars)
        logger.info(f"\n  📝 Formatted Context Preview:")
        logger.info(f"     {result['context'][:500]}...\n")

    logger.info("\n✅ Retrieval testing completed!")


if __name__ == "__main__":
    main()