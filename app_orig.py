import os
import sys
import tempfile
import base64
import gc
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from langchain.callbacks.manager import CallbackManager
from litellm import RateLimitError
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.tools.custom_tool import DocumentSearchTool, FireCrawlWebSearchTool
from config.appconfig import GROQ_API_KEY, FIRECRAWL_API_KEY, SERPER_API_KEY

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  # More reliable path resolution
sys.path.append(os.path.abspath("src"))

# ===========================
#   Constants & Configuration
# ===========================
PDF_UPLOAD_DIR = Path("pdf_uploads") 
# MODEL_MAPPING = {
#     "DeepSeek-R1 (70B)": "groq/deepseek-r1-distill-llama-70b",
#     "Llama-3.3 (70B)": "groq/llama-3.3-70b-versatile"
# }

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
        model="groq/llama-3.3-70b-versatile",
        max_tokens=4096,
        max_retries=5,
        timeout=30,
        metadata={"rate_limit": "10 rpm"},  # Track usage,
        **({"callbacks": callback_manager} if callback_manager else {}),  # Use callbacks instead of callback_manager
        streaming=True,
        verbose=True
    )
@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_kickoff(crew, inputs):
    """Wrapper with rate limit handling"""
    return crew.kickoff(inputs=inputs)


# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks(pdf_tool: Optional[DocumentSearchTool]) -> Crew:
    """Creates a Crew with the given PDF tool (if any) and a web search tool."""
    web_search_tool = SerperDevTool(api_key=SERPER_API_KEY)
    firecrawl_tool = FireCrawlWebSearchTool(api_key=FIRECRAWL_API_KEY)
    groq_llm = get_groq_llm()

    # retriever_agent = Agent(
    #     role="Retrieve relevant information to answer the user query: {query}",
    #     goal=(
    #         "Retrieve the most relevant information from the available sources "
    #         "for the user query: {query}. Always try to use the PDF search tool first. "
    #         "If you are not able to retrieve the information from the PDF search tool, "
    #         "then try to use the web search tool."
    #     ),
    #     backstory=(
    #         "You're a meticulous analyst with a keen eye for detail. "
    #         "You're known for your ability to understand user queries: {query} "
    #         "and retrieve knowledge from the most suitable knowledge base."
    #     ),
    #     verbose=True,
    #     tools=[t for t in [pdf_tool, web_search_tool, firecrawl_tool] if t],
    #     llm=groq_llm,
    #     max_iter=3
    # )
    
    retriever_agent = Agent(
        role="PDF First Retriever",
        goal=(
            "Always use DocumentSearchTool first for {query}. "
            "Only use web search if document search returns empty."
        ),
        tools=[pdf_tool],  # Document tool first
        llm=get_groq_llm(),
        max_iter=2,  # Faster fallback
        allow_delegation=True,
        step_callback=lambda x: gc.collect()  # Memory management
    )
    
    # Add fallback agent
    web_fallback_agent = Agent(
        role="Web Fallback Specialist",
        goal="Handle queries when document search fails",
        tools=[web_search_tool, firecrawl_tool],
        llm=get_groq_llm(),
        allow_delegation=False
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
        max_iter=3,
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
    
    # return Crew(
    #     agents=[retriever_agent, response_synthesizer_agent],
    #     tasks=[retrieval_task, response_task],
    #     process=Process.sequential,  # or Process.hierarchical
    #     verbose=True,
    #     # memory=True,
    #     # cache=True
    # )

    return Crew(
        agents=[retriever_agent, web_fallback_agent, response_synthesizer_agent],
        tasks=[
            Task(
                description="Attempt PDF answer for {query}",
                agent=retriever_agent,
                output_file="pdf_attempt.txt"
            ),
            Task(
                description="Fallback to web search if needed",
                agent=web_fallback_agent,
                output_file="web_fallback.txt",
                context=["pdf_attempt.txt"]  # Only trigger if empty
            ),
            response_task
        ],
        process=Process.hierarchical,
        memory=True,
        verbose=2
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

def extract_cooldown(error_msg):
    """Extract cooldown time from error message"""
    try:
        return float(re.search(r"try again in ([\d.]+)s", error_msg).group(1))
    except:
        return 10  # Default safety

def reset_chat():
    # if 'pdf_tool' in st.session_state:
    #     st.session_state.pdf_tool.clear_cache()
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.pdf_tool = None
    st.session_state.crew = None
    gc.collect()


# ===========================
#   Updated PDF Handling
# ===========================
# def handle_pdf_upload(uploaded_file) -> Optional[Path]:
#     """Process uploaded PDF file and manage UI state."""
#     try:
#         # Check if we already processed this exact file
#         if (st.session_state.get('current_pdf') == uploaded_file.name and 
#             st.session_state.pdf_processed and 
#             st.session_state.pdf_tool is not None):
#             return PDF_UPLOAD_DIR / uploaded_file.name

#         # Clear previous state if new file
#         if st.session_state.get('current_pdf') != uploaded_file.name:
#             reset_chat()

#         # Ensure upload directory exists
#         PDF_UPLOAD_DIR.mkdir(exist_ok=True)
        
#         temp_file = PDF_UPLOAD_DIR / uploaded_file.name
        
#         # Save the uploaded file
#         with temp_file.open("wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Indexing the document
#         with st.spinner("📄 Indexing document... This may take a moment"):
#             st.session_state.pdf_tool = DocumentSearchTool(file_path=str(temp_file))
#             st.session_state.pdf_processed = True
#             st.session_state.current_pdf = uploaded_file.name
#             st.session_state.crew = None  # Reset crew for new document
        
#         st.success("✅ Document successfully indexed and ready for analysis")
#         return temp_file
    
#     except Exception as e:
#         st.error(f"❌ PDF processing error: {e}")
#         return None

# def handle_pdf_upload(uploaded_file) -> Optional[Path]:
#     """Process uploaded PDF file with proper tempfile handling"""
#     try:
#         # Check if we already processed this exact file
#         if (st.session_state.get('current_pdf') == uploaded_file.name and 
#             st.session_state.pdf_processed and 
#             st.session_state.pdf_tool is not None):
#             return PDF_UPLOAD_DIR / uploaded_file.name

#         # Clear previous state if new file
#         if st.session_state.get('current_pdf') != uploaded_file.name:
#             reset_chat()

#         # Create temporary file with proper path handling
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_path = temp_file.name
#             uploaded_file.seek(0)
#             temp_file.write(uploaded_file.getbuffer())
#             temp_file.flush()  # Ensure all data is written

#         # Indexing the document
#         with st.spinner("📄 Indexing document..."):
#             # Pass the actual file path string, not the file object
#             search_tool = DocumentSearchTool(file_path=temp_path)
#             search_tool.index(chunk_size=1024)  # Now calls index with parameters
#             st.session_state.pdf_tool = search_tool
#             st.session_state.pdf_processed = True
#             st.session_state.current_pdf = uploaded_file.name
        
#         # Clean up temporary file after processing
#         try:
#             os.unlink(temp_path)
#         except Exception as e:
#             st.error(f"⚠️ Temporary file cleanup error: {e}")

#         return temp_path
    
#     except Exception as e:
#         st.error(f"📄 Processing error: {str(e)}")
#         return None

def handle_pdf_upload(uploaded_file) -> Optional[Path]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            uploaded_file.seek(0)
            temp_file.write(uploaded_file.getbuffer())

        # PROPER TOOL INITIALIZATION
        st.session_state.pdf_tool = DocumentSearchTool(
            file_path=temp_path  # This is now required
        )
        
        return temp_path
    except Exception as e:
        st.error(f"📄 Processing error: {str(e)}")
        return None


def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF preview in a styled container."""
    base64_pdf = base64.b64encode(file_bytes).decode()
    pdf_display = f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    ">
        <h3 style="margin-top: 0;">Preview: {file_name}</h3>
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="600" 
            style="border-radius: 4px;"
        ></iframe>
    </div>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("📂 Document Management")
    
    # Status section
    st.markdown("")
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

if prompt:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build or reuse the Crew
    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

    # 3. Get the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get context from document search
            context_str = ""
            if st.session_state.pdf_tool:
                try:
                    context = st.session_state.pdf_tool.search(
                        query=prompt, 
                        top_k=3
                    )
                    # Check if context is a list
                    if isinstance(context, list):
                        # Limit context size to prevent token issues
                        context_str = "\n".join([str(c)[:500] for c in context])[:2000]
                    else:
                        context_str = str(context)[:2000]
                except Exception as e:
                    st.error(f"Document search error: {str(e)}")
                    context_str = "Error retrieving context from document."
            
            # Execute with rate limit handling
            with st.spinner("Analyzing..."):
                inputs = {
                    "query": f"{prompt}\n\nRelevant Context: {context_str}",
                }
                
                # Retry logic with backoff
                for attempt in range(3):
                    try:
                        result = st.session_state.crew.kickoff(inputs=inputs).raw
                        break
                    except RateLimitError as e:
                        if attempt == 2:
                            raise
                        wait_time = 5 * (attempt + 1)
                        time.sleep(wait_time)
                    
                    # # Stream response
                    # lines = result.split('\n')
                    # for line in lines:
                    #     full_response += line + "\n"
                    #     message_placeholder.markdown(full_response + "▌")
                    #     time.sleep(0.1)
                    
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": result})

        except RateLimitError as e:
            error_msg = "Our systems are currently at capacity. Please try again in a minute."
            message_placeholder.markdown(f"⚠️ {error_msg}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            time.sleep(10)  # Cooldown period

        except AttributeError as e:
            if "'DocumentSearchTool' object has no attribute 'index'" in str(e):
                error_msg = "Document processing error. Please re-upload your PDF."
                message_placeholder.markdown(f"❌ {error_msg}")
                st.session_state.pdf_tool = None
                reset_chat()
            else:
                raise

        except Exception as e:
            error_msg = "An unexpected error occurred. Please try again."
            message_placeholder.markdown(f"⚠️ {error_msg}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})