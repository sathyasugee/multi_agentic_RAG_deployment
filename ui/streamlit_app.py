"""
Streamlit UI - Multi-Agent RAG System
======================================
Beautiful interactive interface for demonstrating the multi-agent system

FEATURES:
- Chat interface
- Real-time agent routing visualization
- Source citations display
- Agent selection controls
- Conversation history
"""

import streamlit as st
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import config
from agents.orchestrator import Orchestrator
from agents.rag_agent import RAGAgent
from agents.web_search_agent import WebSearchAgent

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=config.get('streamlit.page_title', 'Multi-Agent RAG System'),
    page_icon=config.get('streamlit.page_icon', '🤖'),
    layout=config.get('streamlit.layout', 'wide'),
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }

    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    /* Agent badges */
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }

    .rag-badge {
        background-color: #4caf50;
        color: white;
    }

    .web-badge {
        background-color: #2196f3;
        color: white;
    }

    .both-badge {
        background-color: #ff9800;
        color: white;
    }

    /* Source citations */
    .source-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    /* Metrics */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = None
    if 'web_agent' not in st.session_state:
        st.session_state.web_agent = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0
    if 'agents_loaded' not in st.session_state:
        st.session_state.agents_loaded = False


@st.cache_resource
def load_agents():
    """Load agents once and cache them"""
    with st.spinner("🔄 Loading multi-agent system..."):
        try:
            orchestrator = Orchestrator()
            rag_agent = orchestrator.rag_agent
            web_agent = orchestrator.web_search_agent
            return orchestrator, rag_agent, web_agent, True
        except Exception as e:
            st.error(f"❌ Failed to load agents: {e}")
            return None, None, None, False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_agent_badge(agent_name: str) -> str:
    """Generate HTML badge for agent"""
    if "RAG" in agent_name and "WEB" in agent_name:
        return '<span class="agent-badge both-badge">🤖 BOTH AGENTS</span>'
    elif "RAG" in agent_name:
        return '<span class="agent-badge rag-badge">📚 RAG AGENT</span>'
    elif "WEB" in agent_name:
        return '<span class="agent-badge web-badge">🌐 WEB AGENT</span>'
    else:
        return f'<span class="agent-badge">{agent_name}</span>'


def format_sources(sources, agent_type):
    """Format source citations"""
    if not sources:
        return ""

    html = "<div style='margin-top: 1rem;'>"
    html += "<h4>📚 Sources:</h4>"

    for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
        html += "<div class='source-box'>"

        if 'url' in source and source['url']:
            # Web source
            title = source.get('title', 'Web Source')
            url = source['url']
            html += f"<strong>{i}. {title}</strong><br>"
            html += f"<a href='{url}' target='_blank'>🔗 {url}</a>"
        else:
            # Document source
            preview = source.get('content_preview', '')
            if preview:
                html += f"<strong>{i}. Document Excerpt:</strong><br>"
                html += f"<em>{preview[:200]}...</em>"

            metadata = source.get('metadata', {})
            if metadata.get('row_id'):
                html += f"<br><small>Row ID: {metadata['row_id']}</small>"

        html += "</div>"

    html += "</div>"
    return html


def display_routing_info(routing):
    """Display routing decision visualization"""
    if not routing:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Decision", routing['decision'])

    with col2:
        st.metric("RAG Score", routing.get('rag_score', 0))

    with col3:
        st.metric("Web Score", routing.get('web_score', 0))

    st.info(f"💡 **Reasoning:** {routing['reasoning']}")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main Streamlit application"""

    # Initialize
    init_session_state()

    # Header
    st.title("🤖 Multi-Agent RAG System")
    st.markdown("**Intelligent Question Answering with Automatic Agent Routing**")
    st.markdown("---")

    # Load agents
    if not st.session_state.agents_loaded:
        orchestrator, rag_agent, web_agent, loaded = load_agents()
        if loaded:
            st.session_state.orchestrator = orchestrator
            st.session_state.rag_agent = rag_agent
            st.session_state.web_agent = web_agent
            st.session_state.agents_loaded = True
            st.success("✅ Multi-agent system loaded successfully!")
        else:
            st.error("❌ Failed to load agents. Please check the logs.")
            st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Agent selection
        agent_mode = st.selectbox(
            "Agent Mode",
            ["Auto (Orchestrator)", "RAG Only", "Web Search Only"],
            help="Select which agent(s) to use"
        )

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            top_k = st.slider(
                "Number of documents to retrieve",
                min_value=1,
                max_value=10,
                value=5,
                help="More documents = more context but slower"
            )

            show_routing = st.checkbox(
                "Show routing details",
                value=True,
                help="Display agent routing decision process"
            )

            show_sources = st.checkbox(
                "Show source citations",
                value=True,
                help="Display source documents/URLs"
            )

        # Statistics
        st.markdown("---")
        st.header("📊 Statistics")
        st.metric("Total Queries", st.session_state.total_queries)

        # Example queries
        st.markdown("---")
        st.header("💡 Example Queries")

        examples = {
            "📚 RAG Examples": [
                "What are gaming license requirements?",
                "Tell me about parking regulations",
                "What does Section 5:23 say?"
            ],
            "🌐 Web Search Examples": [
                "Latest AI developments this week",
                "Current weather in New Jersey",
                "Recent gaming industry news"
            ],
            "🤖 Multi-Agent Examples": [
                "Compare NJ gaming laws with federal updates",
                "What are recent changes to gaming regulations?"
            ]
        }

        for category, queries in examples.items():
            with st.expander(category):
                for query in queries:
                    if st.button(query, key=query):
                        st.session_state.example_query = query

        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.header("💬 Chat Interface")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class='user-message'>
                    <strong>👤 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='assistant-message'>
                    <strong>🤖 Assistant:</strong> {get_agent_badge(message.get('agent', 'AGENT'))}<br>
                    {message['content']}
                    {message.get('sources_html', '')}
                </div>
                """, unsafe_allow_html=True)

                # Show routing details if enabled
                if show_routing and message.get('routing'):
                    with st.expander("🎯 View Routing Details"):
                        display_routing_info(message['routing'])

    # Input area
    st.markdown("---")

    # Check for example query
    default_query = ""
    if hasattr(st.session_state, 'example_query'):
        default_query = st.session_state.example_query
        delattr(st.session_state, 'example_query')

    query = st.text_input(
        "Ask a question:",
        value=default_query,
        placeholder="Type your question here...",
        key="query_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🚀 Send", type="primary", use_container_width=True)

    # Process query
    if submit_button and query:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': query
        })

        # Show processing
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()

            try:
                # Execute based on mode
                if agent_mode == "RAG Only":
                    result = st.session_state.rag_agent.answer(query, top_k=top_k)
                elif agent_mode == "Web Search Only":
                    result = st.session_state.web_agent.search_and_answer(query, max_results=top_k)
                else:  # Auto (Orchestrator)
                    result = st.session_state.orchestrator.execute(query)

                processing_time = time.time() - start_time

                # Format sources
                sources_html = ""
                if show_sources and result.get('sources'):
                    sources_html = format_sources(
                        result['sources'],
                        result.get('agent', '')
                    )

                # Add assistant message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'agent': result.get('agent', 'AGENT'),
                    'sources_html': sources_html,
                    'routing': result.get('routing'),
                    'processing_time': processing_time
                })

                # Update stats
                st.session_state.total_queries += 1

                # Show success
                st.success(f"✅ Answered in {processing_time:.2f} seconds")

            except Exception as e:
                st.error(f"❌ Error: {e}")

        # Rerun to update chat
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>
        Multi-Agent RAG System | Built with LangChain, FAISS, Groq, and Streamlit<br>
        Powered by Llama 3.1 and all-MiniLM-L6-v2 embeddings
        </small>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()