import sqlite3
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
import base64
import gc
import time
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.tools.custom_tool import DocumentSearchTool, FireCrawlWebSearchTool
# Replace the import line with
from config.appconfig import (
    GROQ_API_KEY,
    FIRECRAWL_API_KEY,
    SERPER_API_KEY,
    QDRANT_API_KEY,
    QDRANT_LOCATION,
    MODEL
)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  # More reliable path resolution
sys.path.append(os.path.abspath("src"))

# ===========================
#   Constants & Configuration
# ===========================

PDF_UPLOAD_DIR = Path("pdf_uploads")

# ===========================
#   Agent/Task Configuration
# ===========================
def get_groq_llm():
    """Get LLM configuration with proper Groq setup."""
    try:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    except (ImportError, AttributeError):
        callback_manager = None
    
    return ChatGroq(
        temperature=0.3,
        api_key=GROQ_API_KEY,
        model=MODEL,
        max_tokens=4096,
        max_retries=3,
        timeout=30,
        metadata={"rate_limit": "10 rpm"},  # Track usage,
        callbacks=[StreamingStdOutCallbackHandler()],
        # **({"callback_manager": callback_manager} if callback_manager else {}),
        streaming=True,
        verbose=True
    )

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks(pdf_tool: Optional[DocumentSearchTool]) -> Crew:
    """Creates a Crew with the given PDF tool (if any) and a web search tool."""
    web_search_tool = SerperDevTool(api_key=SERPER_API_KEY)
    firecrawl_tool = FireCrawlWebSearchTool(api_key=FIRECRAWL_API_KEY)
    groq_llm = get_groq_llm()

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first. "
            "If you are not able to retrieve the information from the PDF search tool, "
            "then try to use the web search tool."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            "You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base."
        ),
        verbose=True,
        tools=[t for t in [pdf_tool, web_search_tool, firecrawl_tool] if t],
        llm=groq_llm,
        max_iter=2,
        allow_delegation=True,
        step_callback=lambda x: gc.collect()  # Memory management
    )

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            "based on the user query: {query}. If you are not able to retrieve the "
            'information then respond with "I\'m sorry, I couldn\'t find the information '
            'you\'re looking for."'
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses."
        ),
        verbose=True,
        llm=groq_llm,
        max_iter=7,
        allow_delegation=False
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant information from the available "
            "sources for the user query: {query}"
        ),
        expected_output=(
            "The most relevant information in the form of text as retrieved "
            "from the sources."
        ),
        agent=retriever_agent,
        output_file="retrieval_report.md"
    )

    response_task = Task(
        description="Synthesize the final response for the user query: {query}",
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            "from the right source for the user query: {query}. If you are not "
            "able to retrieve the information, then respond with: "
            '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        agent=response_synthesizer_agent,
        output_file="final_response.md"
    )
    
    return Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,  # or Process.hierarchical
        verbose=True,
        # memory=True,
        # cache=True
    )

# ===========================
#   Streamlit Setup
# ===========================
st.set_page_config(
        page_title="Agentic RAG System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

def reset_chat():
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.pdf_tool = None
    st.session_state.crew = None
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF preview in a styled sidebar container."""
    base64_pdf = base64.b64encode(file_bytes).decode()
    pdf_display = f"""
    <div style="
        width: 100%;
        height: 80vh;
        overflow: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 1rem 0;
        box-sizing: border-box;
        background-color: #fafafa;
    ">
        <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">Preview: {file_name}</h3>
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            style="width: 100%; height: calc(100% - 2rem); border: none; border-radius: 4px;"
        ></iframe>
    </div>
    """
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

# ===========================
#   Updated PDF Handling
# ===========================
def handle_pdf_upload(uploaded_file) -> Optional[Path]:
    """Process uploaded PDF file and manage UI state."""
    try:
        # Check if we already processed this exact file
        if (st.session_state.get('current_pdf') == uploaded_file.name and 
            st.session_state.pdf_processed and 
            st.session_state.pdf_tool is not None):
            return PDF_UPLOAD_DIR / uploaded_file.name

        # Clear previous state if new file
        if st.session_state.get('current_pdf') != uploaded_file.name:
            reset_chat()

        # Ensure upload directory exists
        PDF_UPLOAD_DIR.mkdir(exist_ok=True)
        
        temp_file = PDF_UPLOAD_DIR / uploaded_file.name
        
        # Save the uploaded file
        with temp_file.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Indexing the document
        with st.spinner("📄 Indexing document... This may take a moment"):
            st.session_state.pdf_tool = DocumentSearchTool(file_path=str(temp_file))
            st.session_state.pdf_processed = True
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.crew = None  # Reset crew for new document
        
        st.success("✅ Document successfully indexed and ready for analysis")
        return temp_file
    
    except Exception as e:
        st.error(f"❌ PDF processing error: {e}")
        return None

# ===========================
#   Updated Sidebar
# ===========================
with st.sidebar:
    st.header("📂 Document Management")
    
    # Status section
    st.markdown("---")
    st.markdown("**System Status:**")
    status_icon = "🟢" if st.session_state.get("pdf_processed") else "🔴"
    st.markdown(f"{status_icon} Document Processing")
    st.markdown(f"🟢 Model Ready")
    
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload PDF Document", 
        type=["pdf"],
        help="Upload a PDF document for analysis"
    )
    
    if uploaded_file is not None:
        processed_file = handle_pdf_upload(uploaded_file)
        if processed_file:
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    
    # Configuration section
    st.header("⚙️ Settings")
    st.button("🧹 Clear Chat History", on_click=reset_chat)


# ===========================
#   Main Chat Interface
# ===========================

st.markdown("""# Intelligent Document Analysis System""", unsafe_allow_html=True)

st.markdown(
        """
        **Welcome to the AI-Powered Document Retrieval Assistant 👋**
                
        This app allows you to interact with an AI-powered assistant and upload documents and retrieve information from them.

        """
    )

# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your PDF...")

# ===========================
#   Updated Chat Interface
# ===========================
if prompt:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build or reuse the Crew (only once after PDF is loaded)
    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

    # 3. Get the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get the complete response first
            with st.spinner("Thinking..."):
                inputs = {"query": prompt}
                result = st.session_state.crew.kickoff(inputs=inputs).raw
            
            # Split by lines first to preserve code blocks and other markdown
            lines = result.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:  # Don't add newline to the last line
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.15)  # Adjust the speed as needed
            
            # Show the final response without the cursor
            message_placeholder.markdown(full_response)
            
            # Save assistant's message to session
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            # Handle rate limit errors specifically
            if "RateLimitError" in str(e) or "rate_limit_exceeded" in str(e):
                error_msg = "⚠️ Our systems are currently experiencing high demand. Please try again in a few moments."
            else:
                error_msg = "⚠️ An unexpected error occurred. Please try again later."
            
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})