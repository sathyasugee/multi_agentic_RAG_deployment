"""
FastAPI Application - Multi-Agent RAG System
=============================================
Production-ready REST API for the multi-agent system

ENDPOINTS:
- POST /query - Main query endpoint
- POST /rag - RAG agent only
- POST /web-search - Web search agent only
- GET /health - Health check
- GET / - API documentation redirect
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from loguru import logger
import time

# Import from parent directory
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import config
from agents.orchestrator import Orchestrator
from agents.rag_agent import RAGAgent
from agents.web_search_agent import WebSearchAgent


# ============================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="User's question", min_length=3, max_length=500)
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1, le=10)
    use_web_search: Optional[bool] = Field(None, description="Force use of web search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are NJ gaming license requirements?",
                "top_k": 5
            }
        }


class Source(BaseModel):
    """Source information model"""
    title: Optional[str] = None
    url: Optional[str] = None
    content_preview: Optional[str] = None
    metadata: Optional[Dict] = None


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source citations")
    num_sources: int = Field(..., description="Number of sources used")
    agent: str = Field(..., description="Agent(s) that handled the query")
    routing: Optional[Dict] = Field(None, description="Routing decision details")
    processing_time: float = Field(..., description="Query processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "According to Section 5:23...",
                "sources": [{"content_preview": "Section 5:23 states..."}],
                "num_sources": 3,
                "agent": "RAG_AGENT",
                "processing_time": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents_loaded: bool
    vector_store_loaded: bool


# ============================================================
# FASTAPI APPLICATION
# ============================================================

# Initialize FastAPI app
app = FastAPI(
    title=config.get('api.title', 'Multi-Agent RAG API'),
    version=config.get('api.version', '1.0.0'),
    description="""
    Multi-Agent RAG System API with intelligent routing.

    ## Features
    - **Intelligent Routing**: Automatically selects the best agent(s)
    - **RAG Agent**: Answers from document knowledge base
    - **Web Search Agent**: Retrieves current information from internet
    - **Multi-Agent Fusion**: Combines both sources when needed

    ## Usage
    Send POST request to `/query` with your question.
    The system will automatically route to the appropriate agent(s).
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL STATE (Loaded once at startup)
# ============================================================

orchestrator = None
rag_agent = None
web_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global orchestrator, rag_agent, web_agent

    logger.info("🚀 Starting FastAPI server...")
    logger.info(f"📍 API Host: {config.get('api.host', '0.0.0.0')}")
    logger.info(f"📍 API Port: {config.get('api.port', 8000)}")

    try:
        # Initialize orchestrator (loads all agents)
        logger.info("🔄 Loading multi-agent system...")
        orchestrator = Orchestrator()
        rag_agent = orchestrator.rag_agent
        web_agent = orchestrator.web_search_agent

        logger.info("✅ All agents loaded successfully")
        logger.info("🎉 FastAPI server ready!")

    except Exception as e:
        logger.error(f"❌ Failed to load agents: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down FastAPI server...")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns system status and agent availability
    """
    return HealthResponse(
        status="healthy" if orchestrator else "initializing",
        version=config.get('api.version', '1.0.0'),
        agents_loaded=orchestrator is not None,
        vector_store_loaded=rag_agent.retriever.vectorstore is not None if rag_agent else False
    )


@app.post("/query", response_model=QueryResponse, tags=["Main"])
async def query(request: QueryRequest):
    """
    **Main Query Endpoint - Intelligent Multi-Agent Routing**

    This endpoint automatically routes your query to the best agent(s):
    - Questions about documents → RAG Agent
    - Questions about recent/current events → Web Search Agent
    - Questions requiring both → Multi-Agent Synthesis

    ## Example Queries:
    - "What are gaming license requirements?" → RAG Agent
    - "What are recent AI developments?" → Web Search Agent
    - "Compare NJ laws with federal updates" → Both Agents

    ## Parameters:
    - **query**: Your question (required)
    - **top_k**: Number of documents to retrieve (optional, default: 5)
    - **use_web_search**: Force web search usage (optional)
    """
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agents not initialized. Please try again."
        )

    try:
        start_time = time.time()
        logger.info(f"📥 Received query: '{request.query}'")

        # Execute orchestrator
        result = orchestrator.execute(request.query)

        processing_time = time.time() - start_time
        logger.info(f"✅ Query processed in {processing_time:.2f}s")

        # Format sources
        sources = []
        for src in result.get('sources', []):
            if isinstance(src, dict):
                sources.append(Source(**src))
            else:
                sources.append(Source(content_preview=str(src)))

        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            num_sources=result['num_sources'],
            agent=result['agent'],
            routing=result.get('routing'),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/rag", response_model=QueryResponse, tags=["Agents"])
async def query_rag(request: QueryRequest):
    """
    **RAG Agent Only - Document-Based Answers**

    Forces the use of RAG agent (vector database search).
    Use this when you specifically want answers from your documents.

    ## Best for:
    - Specific code/regulation questions
    - Historical information
    - Known document content
    """
    if not rag_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG agent not initialized"
        )

    try:
        start_time = time.time()
        logger.info(f"📚 RAG query: '{request.query}'")

        # Execute RAG agent directly
        result = rag_agent.answer(request.query, top_k=request.top_k)

        processing_time = time.time() - start_time

        # Format sources
        sources = [Source(**src) for src in result.get('sources', [])]

        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            num_sources=result['num_sources'],
            agent=result['agent'],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"❌ RAG error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG agent error: {str(e)}"
        )


@app.post("/web-search", response_model=QueryResponse, tags=["Agents"])
async def query_web(request: QueryRequest):
    """
    **Web Search Agent Only - Internet Research**

    Forces the use of web search agent (DuckDuckGo search).
    Use this when you need current/external information.

    ## Best for:
    - Recent news/updates
    - Current events
    - External information not in documents
    """
    if not web_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search agent not initialized"
        )

    try:
        start_time = time.time()
        logger.info(f"🌐 Web search query: '{request.query}'")

        # Execute web search agent directly
        result = web_agent.search_and_answer(request.query, max_results=request.top_k)

        processing_time = time.time() - start_time

        # Format sources
        sources = [Source(**src) for src in result.get('sources', [])]

        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            num_sources=result['num_sources'],
            agent=result['agent'],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"❌ Web search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Web search agent error: {str(e)}"
        )


@app.get("/stats", tags=["System"])
async def get_stats():
    """
    Get system statistics

    Returns information about the loaded agents and configuration
    """
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )

    return {
        "agents": {
            "rag_agent": {
                "name": rag_agent.get_name(),
                "model": config.llm_model,
                "embedding_model": config.embedding_model,
                "top_k": config.top_k
            },
            "web_search_agent": {
                "name": web_agent.get_name(),
                "model": config.llm_model,
                "max_results": config.get('agents.max_search_results', 3)
            }
        },
        "routing": {
            "rag_keywords": config.get('agents.routing.rag_keywords', []),
            "web_keywords": config.get('agents.routing.web_keywords', [])
        },
        "config": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "temperature": config.get('llm.temperature', 0.3)
        }
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# ============================================================
# RUN SERVER (for development)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "fastapi_app:app",
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 8000),
        reload=config.get('api.reload', True),
        log_level="info"
    )