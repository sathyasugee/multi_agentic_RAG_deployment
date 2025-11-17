"""
Configuration Management for Multi-Agent RAG System
Loads settings from config.yaml and .env files
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration management"""

    def __init__(self, config_path: str = "config.yaml"):
        # Always resolve path relative to project root
        if not Path(config_path).is_absolute():
            # Get project root (parent of src/)
            project_root = Path(__file__).parent.parent
            self.config_path = project_root / config_path
        else:
            self.config_path = Path(config_path)

        self.config = self._load_yaml()
        self._setup_logging()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise

    def _setup_logging(self):
        """Configure logging based on config"""
        log_config = self.config.get('logging', {})
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format')
        )

    def _validate_config(self):
        """Validate critical configuration and environment variables"""
        # Check API key
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("⚠ GROQ_API_KEY not set in .env file")

        # Get project root (parent of src/)
        project_root = Path(__file__).parent.parent

        # Create data directories relative to project root
        for key in ['raw_path', 'processed_path', 'vector_store_path']:
            path_str = self.config['data'][key]
            # Make absolute path from project root
            path = project_root / path_str.lstrip('./')
            path.parent.mkdir(parents=True, exist_ok=True)

            # Update config with absolute path
            self.config['data'][key] = str(path)

        logger.info(f"✓ Data paths resolved from: {project_root}")

    # ========== Property Accessors ==========

    @property
    def groq_api_key(self) -> str:
        """Get Groq API key from environment"""
        return os.getenv("GROQ_API_KEY", "")

    @property
    def embedding_model(self) -> str:
        """Get embedding model name"""
        return self.config['embedding']['model_name']

    @property
    def llm_model(self) -> str:
        """Get LLM model name"""
        return self.config['llm']['model']

    @property
    def chunk_size(self) -> int:
        """Get text chunk size"""
        return self.config['chunking']['chunk_size']

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap size"""
        return self.config['chunking']['chunk_overlap']

    @property
    def top_k(self) -> int:
        """Get number of retrieval results"""
        return self.config['retrieval']['top_k']

    @property
    def vector_store_path(self) -> Path:
        """Get vector store directory path"""
        return Path(self.config['data']['vector_store_path'])

    @property
    def raw_data_path(self) -> Path:
        """Get raw data file path"""
        return Path(self.config['data']['raw_path'])

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        Example: config.get('llm.temperature') returns 0.3
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, llm={self.llm_model})"


# Global config instance (singleton pattern)
config = Config()

if __name__ == "__main__":
    # Test configuration loading
    print(f"🔧 Configuration Test")
    print(f"LLM Model: {config.llm_model}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Top-K Retrieval: {config.top_k}")
    print(f"Vector Store: {config.vector_store_path}")
    print(f"API Key Set: {'✓' if config.groq_api_key else '✗'}")