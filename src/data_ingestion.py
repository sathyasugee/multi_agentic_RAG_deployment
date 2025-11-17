"""
Data Ingestion Pipeline for Multi-Agent RAG System
===================================================
Clean, production-ready pipeline that:
1. Loads CSV data
2. Chunks text intelligently
3. Generates embeddings
4. Stores in FAISS vector database
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
from loguru import logger

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import config


class DataIngestion:
    """Handles data loading, chunking, and vector storage"""

    def __init__(self):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None

        # Initialize everything directly in __init__
        logger.info("🔧 Initializing Data Ingestion Pipeline...")

        # Setup embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✓ Loaded embeddings: {self.config.embedding_model}")

        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"✓ Text splitter: {self.config.chunk_size} chars, {self.config.chunk_overlap} overlap")

    def load_csv(self, csv_path: str = None) -> pd.DataFrame:
        """Load CSV data"""
        path = Path(csv_path) if csv_path else self.config.raw_data_path

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"✓ Loaded CSV: {len(df)} rows, {list(df.columns)}")
        return df

    def prepare_documents(self, df: pd.DataFrame, text_column: str) -> List[Document]:
        """
        Convert DataFrame to LangChain Documents with metadata

        Args:
            df: Input dataframe
            text_column: Column name containing text to embed

        Returns:
            List of Document objects
        """
        documents = []

        for idx, row in df.iterrows():
            # Extract text
            text = str(row[text_column])

            # Skip empty or very short texts
            if len(text.strip()) < self.config.get('chunking.min_chunk_size', 100):
                continue

            # Create metadata (all other columns)
            metadata = {
                'row_id': idx,
                'source': 'nj_admin_code'
            }

            # Add all columns as metadata (except text column)
            for col in df.columns:
                if col != text_column:
                    metadata[col] = str(row[col])

            # Create LangChain Document
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        logger.info(f"✓ Created {len(documents)} documents")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"✓ Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from chunks"""
        logger.info("🔄 Generating embeddings and creating vector store...")

        # Create FAISS index
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        logger.info(f"✓ Vector store created with {len(chunks)} embeddings")
        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS, path: str = None):
        """Save FAISS index to disk"""
        save_path = Path(path) if path else self.config.vector_store_path
        save_path.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(str(save_path))
        logger.info(f"✓ Vector store saved to: {save_path}")

    def load_vectorstore(self, path: str = None) -> FAISS:
        """Load existing FAISS index"""
        load_path = Path(path) if path else self.config.vector_store_path

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found: {load_path}")

        vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"✓ Vector store loaded from: {load_path}")
        return vectorstore

    def run_pipeline(self, csv_path: str = None, text_column: str = None) -> FAISS:
        """
        Complete ingestion pipeline

        Args:
            csv_path: Path to CSV file
            text_column: Column name with text to embed

        Returns:
            FAISS vectorstore
        """
        logger.info("🚀 Starting Data Ingestion Pipeline")

        # Step 1: Load CSV
        df = self.load_csv(csv_path)

        # Step 2: Auto-detect text column if not provided
        if text_column is None:
            # Find column with longest average text
            text_lengths = {col: df[col].astype(str).str.len().mean()
                            for col in df.columns}
            text_column = max(text_lengths, key=text_lengths.get)
            logger.info(f"ℹ Auto-detected text column: '{text_column}'")

        # Step 3: Prepare documents
        documents = self.prepare_documents(df, text_column)

        # Step 4: Chunk documents
        chunks = self.chunk_documents(documents)

        # Step 5: Create vector store
        vectorstore = self.create_vectorstore(chunks)

        # Step 6: Save to disk
        self.save_vectorstore(vectorstore)

        self.vectorstore = vectorstore
        logger.info("✅ Pipeline completed successfully!")
        return vectorstore


# ============================================================
# CLI Interface for Testing
# ============================================================

def main():
    """Run data ingestion from command line"""
    import sys

    # Initialize pipeline
    ingestion = DataIngestion()

    # Check if CSV exists
    csv_path = config.raw_data_path
    if not csv_path.exists():
        logger.error(f"❌ CSV file not found at: {csv_path}")
        logger.info(f"💡 Please place your CSV file at: {csv_path}")
        sys.exit(1)

    # Run pipeline
    try:
        vectorstore = ingestion.run_pipeline()

        # Test retrieval
        logger.info("\n🧪 Testing retrieval...")
        test_query = "gaming license"
        results = vectorstore.similarity_search(test_query, k=3)

        logger.info(f"Query: '{test_query}'")
        for i, doc in enumerate(results, 1):
            logger.info(f"\n  Result {i}:")
            logger.info(f"  Text: {doc.page_content[:100]}...")
            logger.info(f"  Metadata: {doc.metadata}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()