"""
TLDR Summary Generator - Streamlit Web Interface
"""
version = "1.0.37"

import os
import sys
import math
import asyncio
import traceback
from pathlib import Path
from typing import List

import streamlit as st
#from streamlit.runtime.uploaded_file_manager import UploadedFile
# from streamlit.delta_generator import DeltaGenerator

# Add the parent directory to path to import tldr
sys.path.append(str(Path(__file__).parent))

# Import tldr components
from tldr.core import TldrEngine

# Page config
st.set_page_config(
    page_title="TLDR Summary Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin: 0.25rem 0;
    }
    .stTextArea>div>div>textarea {
        min-height: 300px;
    }
    .file-uploader {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .file-uploader:hover {
        border-color: #2563eb;
    }
    .file-list {
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .file-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .file-item:last-child {
        border-bottom: none;
    }
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #f8fafc;
        padding: 0.5rem 1rem;
        border-top: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #64748b;
    }
    .process-button button {
        font-weight: bold !important;
        font-size: 1.2rem !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


class TldrUI:
    def __init__(self):
        self.status = "Ready"
        self.processing = False

    def log_message(self, message, level="INFO"):
        """Helper method to log messages to both console and Streamlit interface"""
        formatted_message = f"{level}: {message}"
        print(f"TLDR: {formatted_message}")  # Console logging
        
        # Log to TldrEngine logger if available
        if hasattr(self, 'tldr') and hasattr(self.tldr, 'logger'):
            if level.upper() == "ERROR":
                self.tldr.logger.error(formatted_message)
            elif level.upper() == "WARNING":
                self.tldr.logger.warning(formatted_message)
            else:
                self.tldr.logger.info(formatted_message)
    
    def start_processing_status(self, message: str):
        """Helper method to update processing status"""
        self.log_message(message)
        self.status = "Processing..."
        self.processing = True
        st.session_state.status = self.status
        st.session_state.processing = self.processing

    def end_processing_status(self):
        """Reset session status and update session state"""
        self.log_message("Completed successfully")

        # Restore session to ready state
        self.status = "Ready"
        self.processing = False
        st.session_state.status = self.status
        st.session_state.processing = self.processing
        st.session_state.total_spend = self.round_up(self.tldr.total_spend)
        st.session_state.input_token_count = self.tldr.total_input_tokens
        st.session_state.output_token_count = self.tldr.total_output_tokens

    def session_error(self, error_message: str):
        error_details = traceback.format_exc()
        self.status = "Error during processing"
        self.log_message(f"Error: {error_message}", "ERROR")
        self.log_message(f"Error details: {error_details}", "ERROR")
        st.error(f"An error occurred: {error_message}\n\nError details:\n{error_details}")
        st.stop()

    @staticmethod
    def round_up(num, to=0.01):
        """Round up to the nearest specified decimal"""
        return round(num + to / 2, -int(math.log10(to)))

    async def execute_session_process(self, log_message, function, **args):
        """Generic execution of an async function with processing status updates"""
        self.start_processing_status(log_message)
        
        try:
            await function(**args)
        except Exception as e:
            self.session_error(str(e))

        self.end_processing_status()

    def process_files(self, files) -> List[dict]:
        """Process uploaded files and return file info"""
        file_info = []
        for file in files:
            file_path = os.path.join(self.tldr.output_directory, file.name)
            # Get file size before writing
            file_size = len(file.getvalue())
            # Write file content
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_info.append(
                {
                    "name": file.name,
                    "size": f"{file_size / 1024:.1f} KB",
                    "source": file_path,
                }
            )
        return file_info

    def document_uploader(self, header: str, key: str, description: str = ""):
        """Document uploader with logging of uploaded files"""
        st.subheader(header)

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            disabled=self.processing,
            help=description,
            accept_multiple_files=True,
            key=key,
        )
        
        # Log the uploaded files
        if uploaded_files and len(uploaded_files) > 0:
            file_names = ", ".join([f"'{file.name}'" for file in uploaded_files])
            self.log_message(f"Uploaded {len(uploaded_files)} file(s) to {key}: {file_names}")
        
        return uploaded_files


async def run_tldr_streamlit():

    # Initialize console logging from the start
    print(f"TLDR v{version}: Starting Streamlit application")
    print("TLDR: Console logging is enabled - all operations will be logged to terminal")

    # Initialize session state
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "added_context" not in st.session_state:
        st.session_state.added_context = None
    if "executive_summary" not in st.session_state:
        st.session_state.executive_summary = None
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "polished_summary" not in st.session_state:
        st.session_state.polished_summary = None
    if "summarized" not in st.session_state:
        st.session_state.summarized = False
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = None
    if "input_token_count" not in st.session_state:
        st.session_state.input_token_count = 0
    if "output_token_count" not in st.session_state:
        st.session_state.output_token_count = 0
    if "total_spend" not in st.session_state:
        st.session_state.total_spend = 0.0
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None

    # Initialize TLDR classes if first run
    if "tldr_ui" not in st.session_state:
        st.session_state.tldr_ui = TldrUI()
        st.session_state.tldr_ui.tldr = TldrEngine()
        
        # Log initialization success
        print("TLDR: TldrEngine initialized successfully")
        print("TLDR: StreamlitLogHandler is ready for log capture")
        
        # Verify logger setup
        if hasattr(st.session_state.tldr_ui.tldr, 'logger'):
            logger = st.session_state.tldr_ui.tldr.logger
            print(f"TLDR: Logger has {len(logger.handlers)} handlers configured")
            
            # Make sure we have at least console and streamlit handlers
            handler_types = [type(h).__name__ for h in logger.handlers]
            print(f"TLDR: Handler types: {handler_types}")
            
            # Test logging
            st.session_state.tldr_ui.log_message("TLDR system initialized and ready for use")
        else:
            print("TLDR WARNING: No logger found on TldrEngine")
                
    tldr_ui = st.session_state.tldr_ui

    # Handle refined query and initial context updates
    if "refined_query" in st.session_state:
        st.session_state.user_query = st.session_state.refined_query
        del st.session_state.refined_query
    if "initial_context" in st.session_state:
        st.session_state.added_context = st.session_state.initial_context
        del st.session_state.initial_context

    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")

        # Select options
        st.subheader("Options")
        tone = st.selectbox("Polished summary tone", ["stylized", "formal"], index=0,
            help="The tone of the polished summary (Stylized follows writing style of certain prominent authors).")
        context_size = st.selectbox(
            "Context window", ["small", "medium", "large"], index=1, 
                help="The context window size for the summarization and research effort.")
        # Status
        st.divider()
        st.subheader("Status")
        st.text(f"System: {tldr_ui.status}")
        st.text(f"Input Token Count: {st.session_state.input_token_count}")
        st.text(f"Output Token Count: {st.session_state.output_token_count}")
        st.text(f"Approx. Total Spend: ${st.session_state.total_spend}")

    # Main content area
    st.title(f"📝 Too Long; Didn't Read (TLDR) - v{version}")
    st.markdown("Generate concise summaries, research knowledge gaps, and synthesize information from your documents with AI assistance")

    # Create two main columns directly
    col1, col2 = st.columns([1, 1], gap="medium")

    # Left column - Document input and controls
    with col1:
        # Upload files
        documents = tldr_ui.document_uploader(
            "Target Documents", 
            "document_uploader", 
            "Selected documents to focus on for summaries and research")
        context = tldr_ui.document_uploader(
            "Additional Context", 
            "context_uploader", 
            "Documents to provide supplementary context")

        # Process uploaded files
        if documents is not None or context is not None:
            st.markdown('<div class="process-button">', unsafe_allow_html=True)
            process_clicked = st.button("Process References",
                help="Process the documents you want to summarize and research.")
            st.markdown("</div>", unsafe_allow_html=True)

            if process_clicked:
                with st.spinner("Processing documents and context..."):
                    st.session_state.documents = tldr_ui.process_files(documents)
                    input_files = [f["source"] for f in st.session_state.documents]
                    if context is not None:
                        st.session_state.context = tldr_ui.process_files(context)
                        context_files = [f["source"] for f in st.session_state.context]
                    else:
                        context_files = None

                    # Collect all content
                    await tldr_ui.execute_session_process(
                        log_message=f"Starting to load content from {len(input_files)} files...",
                        function=tldr_ui.tldr.load_all_content,
                        input_files=input_files,
                        context_files=context_files,
                        context_size=context_size,
                    )
                    # And update the session state
                    for doc in st.session_state.documents:
                        doc["content"] = tldr_ui.tldr.content[doc["source"]]

                    if context_files is not None:
                        st.session_state.added_context = tldr_ui.tldr.added_context

        # Query input
        st.subheader("Focused Query")
        st.text_area(
            "What would you like to know from these documents?",
            height=70,
            key="user_query",
        )
        query_col1, query_col2 = st.columns(2)

        # Query refine button, display refined text once returned
        with query_col1:
            if st.button(
                "Refine Query",
                disabled=st.session_state.user_query is None or tldr_ui.processing is True,
            ):
                with st.spinner("Refining query..."):
                    await tldr_ui.execute_session_process(
                        log_message=f"Refining query: {st.session_state.user_query}",
                        function=tldr_ui.tldr.refine_user_query,
                        query=st.session_state.user_query,
                    )
                    st.session_state.refined_query = tldr_ui.tldr.query
                    st.rerun()

        with query_col2:
            if st.button(
                "Search for Context",
                disabled=st.session_state.user_query is None or tldr_ui.processing is True or st.session_state.documents is None,
            ):
                with st.spinner("Searching docs and web for more context (please wait)..."):
                    await tldr_ui.execute_session_process(
                        log_message=f"Searching for context: {st.session_state.user_query}",
                        function=tldr_ui.tldr.initial_context_search,
                        context_size=context_size,
                    )
                    st.session_state.initial_context = tldr_ui.tldr.added_context
                    st.rerun()

        # Display file listdoc
        if st.session_state.documents is None:
            st.info("Upload documents first to view their contents.")
        else:
            st.subheader("Uploaded Files")

            # Display file list with selection
            for doc in st.session_state.documents:
                with st.container():
                    file_name_col, del_btn_col = st.columns([4, 1])
                    with file_name_col:
                        # Make file name clickable for selection
                        if st.button(
                            doc["source"],
                            key=f"select_{doc['source']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_doc = doc
                    with del_btn_col:
                        if st.button("🗑️", key=f"del_{doc['source']}"):
                            st.session_state.documents = [
                                d
                                for d in st.session_state.documents
                                if d["source"] != doc["source"]
                            ]
                            if (
                                st.session_state.selected_doc
                                and st.session_state.selected_doc["source"]
                                == doc["source"]
                            ):
                                st.session_state.selected_doc = None

            # Display selected document content 
            if st.session_state.selected_doc is None:
                st.info("Select a document to view its content.")
            elif st.session_state.selected_doc is not None:
                # Display content in a scrollable text area
                if "content" in st.session_state.selected_doc and st.session_state.selected_doc["content"]:
                    content = st.session_state.selected_doc["content"]
                    if isinstance(content, str):
                        content_html = content.replace("\n", "<br>")
                        st.markdown(
                            f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                            f'{content_html}'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif isinstance(content, dict):
                        # If content is a dictionary, try to find the actual text content
                        text_content = None
                        for key in ["text", "content", "raw_text", "body"]:
                            if key in content and isinstance(content[key], str):
                                text_content = content[key]
                                break
                        
                        if text_content:
                            content_html = text_content.replace("\n", "<br>")
                            st.markdown(
                                f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                                f'{content_html}'
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Show the structure for debugging
                            st.write("Content structure:")
                            st.json(content)
                    else:
                        st.warning("Document content is not in a readable format.")

    # Right column - Summaries and actions
    with col2:
        # Generate initial summaries
        if st.button("Generate Reference Summaries", 
                     disabled=st.session_state.documents is None or tldr_ui.processing is True,
                     help="Summarizes the initial set of documents to create a baseline."):
            with st.spinner("Summarizing documents..."):

                # Generate document summaries
                await tldr_ui.execute_session_process(
                    log_message="Summarizing documents...",
                    function=tldr_ui.tldr.summarize_resources,
                    context_size=context_size,
                )
                # And update the session state
                for doc in st.session_state.documents:
                    doc["summary"] = tldr_ui.tldr.content[doc["source"]]["summary"]
                st.session_state.summarized = True

        # Display selected document content
        if st.session_state.summarized is True:
            st.subheader("Document Summaries")
            st.text("Select a document from the list of uploaded documents to view its summary.")
            # Display summary in a scrollable text area
            summary = st.session_state.selected_doc.get(
                    "summary", "No summary available"
                )
            st.markdown(
                f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                f"{summary}"
                "</div>",
                unsafe_allow_html=True,
                )
        else:
            st.info("Generate reference summaries first to view document summaries.")
           
        st.subheader("Actions")
        # Create three columns for buttons
        action_col1, action_col2, action_col3 = st.columns(3)

        # Research
        with action_col1:
            if st.button(
                "Research",
                disabled=st.session_state.documents is None
                or st.session_state.summarized is False
                or tldr_ui.processing is True,
                help="Apply external research to the compiled document summaries",
            ):
                with st.spinner("Researching knowledge gaps (please wait)..."):
                    await tldr_ui.execute_session_process(
                        log_message="Researching knowledge gaps...",
                        function=tldr_ui.tldr.apply_research,
                        context_size=context_size,
                    )

        # Synthesis
        with action_col2:
            if st.button(
                "Synthesize",
                disabled=st.session_state.documents is None
                or st.session_state.summarized is False
                or tldr_ui.processing is True,
                help="Synthesize summaries, research, and new added context",
            ):
                with st.spinner(
                    "Synthesizing summaries, research, and new added context..."
                ):
                    await tldr_ui.execute_session_process(
                        log_message="Synthesizing summaries, research, and new added context...",
                        function=tldr_ui.tldr.integrate_summaries,
                        context_size=context_size,
                    )
                    st.session_state.executive_summary = tldr_ui.tldr.executive_summary
    
        # Polish
        with action_col3:
            if st.button(
                "Polish",
                disabled=st.session_state.executive_summary is None
                or st.session_state.summarized is False
                or tldr_ui.processing is True,
                help="Polish the finalized summary",
            ):
                with st.spinner("Polishing finalized summary..."):
                    await tldr_ui.execute_session_process(
                        log_message="Polishing finalized summary...",
                        function=tldr_ui.tldr.polish_response,
                        tone=tone, context_size=context_size
                    )
                    st.session_state.polished_summary = tldr_ui.tldr.polished_summary

        st.subheader("TLDR Text")
        # Summary tabs
        tabs = st.tabs(
            ["Added Context", "Research Results", "Executive Summary", "Polished Summary"]
        )
        tab_names = ['added_context', 'research_context', 'executive_summary', 'polished_summary']
        
        # Update active tab in session state
        for i, tab in enumerate(tabs):
            with tab:
                # This will run when the tab is active
                st.session_state.active_tab = tab_names[i]

        # Summary content
        # Initialize edit mode state if not exists
        if 'edit_mode' not in st.session_state:
            st.session_state.edit_mode = {
                'added_context': False,
                'research_context': False,
                'executive_summary': False,
                'polished_summary': False
            }
        
        # Initialize edit buffers if not exists
        if 'edit_buffers' not in st.session_state:
            st.session_state.edit_buffers = {
                'added_context': "",
                'research_context': "",
                'executive_summary': "",
                'polished_summary': ""
            }
        
        # Update edit buffers from TldrEngine if not in edit mode
        if hasattr(tldr_ui, 'tldr'):
            if not st.session_state.edit_mode['added_context'] and hasattr(tldr_ui.tldr, 'added_context'):
                st.session_state.edit_buffers['added_context'] = tldr_ui.tldr.added_context or ""
            if not st.session_state.edit_mode['research_context'] and hasattr(tldr_ui.tldr, 'research_context'):
                st.session_state.edit_buffers['research_context'] = tldr_ui.tldr.research_context or ""
            if not st.session_state.edit_mode['executive_summary'] and hasattr(tldr_ui.tldr, 'executive_summary'):
                st.session_state.edit_buffers['executive_summary'] = tldr_ui.tldr.executive_summary or ""
            if not st.session_state.edit_mode['polished_summary'] and hasattr(tldr_ui.tldr, 'polished_summary'):
                st.session_state.edit_buffers['polished_summary'] = tldr_ui.tldr.polished_summary or ""
        
        # Tab contents
        tab_contents = [
            {
                'title': "Additional context provided during research",
                'key': 'added_context',
                'help_text': "Edit additional context"
            },
            {
                'title': "Research results from external sources",
                'key': 'research_context',
                'help_text': "Edit research context"
            },
            {
                'title': "Executive summary of combined document summaries and research results",
                'key': 'executive_summary',
                'help_text': "Edit executive summary"
            },
            {
                'title': "Polished executive summary with improved formatting and tone",
                'key': 'polished_summary',
                'help_text': "Edit polished summary"
            }
        ]

        # Create tab contents
        for i, tab in enumerate(tabs):
            with tab:
                tab_info = tab_contents[i]
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"✏️ Edit", key=f"edit_{tab_info['key']}"):
                        st.session_state.edit_mode[tab_info['key']] = True
                with col2:
                    if st.session_state.edit_mode[tab_info['key']] and st.button("💾 Save", key=f"save_{tab_info['key']}"):
                        st.session_state.edit_mode[tab_info['key']] = False
                        setattr(tldr_ui.tldr, tab_info['key'], st.session_state.edit_buffers[tab_info['key']])

                st.text_area(
                    tab_info['title'],
                    value=st.session_state.edit_buffers[tab_info['key']],
                    key=f"{tab_info['key']}_area",
                    disabled=not st.session_state.edit_mode[tab_info['key']],
                    on_change=lambda k=tab_info['key']: st.session_state.edit_buffers.update({
                        k: st.session_state[f"{k}_area"]
                    })
                )
        st.write("")

        st.subheader("Handle Output")
        # Create three columns for buttons
        output_col1, output_col2, output_col3 = st.columns(3)

        with output_col1:
            # Determine which tab is active and get its content
            active_tab = st.session_state.get('active_tab', 'executive_summary')
            tab_content = st.session_state.edit_buffers.get(active_tab, "")
            
            if st.button(
                "📋 Copy to Clipboard",
                key="copy_btn_hidden",
                disabled=not tab_content.strip(),
                help="Copy current tab's content to clipboard",
            ):
                st.session_state.clipboard = tab_content
                st.toast("Copied to clipboard!")

        with output_col2:
            if st.button(
                "💾 Save Executive Summary as PDF",
                key="executive_pdf_btn_hidden",
                disabled=st.session_state.executive_summary is None,
                help="Save summary as PDF",
            ):
                with st.spinner("Generating PDF..."):
                    await tldr_ui.execute_session_process(
                        log_message="Saving to PDF...",
                        function=tldr_ui.tldr.save_to_pdf,
                        polished=False,
                    )
                st.toast("PDF saved!")

        with output_col3:
            if st.button(
                "💾 Save Polished Summary as PDF",
                key="polished_pdf_btn_hidden",
                disabled=st.session_state.polished_summary is None,
                help="Save summary as PDF",
            ):
                with st.spinner("Generating PDF..."):
                    await tldr_ui.execute_session_process(
                        log_message="Saving to PDF...",
                        function=tldr_ui.tldr.save_to_pdf,
                        polished=True,
                    )
                st.toast("PDF saved!")


    # Status and Output section
    with st.expander("🔍 Processing Logs", expanded=False):
        # Initialize session state for logs if needed
        if 'output_lines' not in st.session_state:
            st.session_state.output_lines = []
        
        # Clear logs button
        if st.button("Clear Logs"):
            st.session_state.output_lines = []
        
        # Try to get logs from different possible sources
        new_logs = []
        
        # Method 1: Check for StreamlitLogHandler if it exists
        if hasattr(tldr_ui.tldr, 'streamlit_handler') and tldr_ui.tldr.streamlit_handler is not None:
            try:
                logs = tldr_ui.tldr.streamlit_handler.get_logs()
                if logs:
                    new_logs.extend(logs)
                    # Clear the handler logs after reading to prevent duplication
                    tldr_ui.tldr.streamlit_handler.clear_logs()
            except Exception as e:
                print(f"Error reading streamlit handler logs: {e}")
        
        # Method 2: Check for any log records from the logger directly
        if hasattr(tldr_ui.tldr, 'logger') and hasattr(tldr_ui.tldr.logger, 'handlers'):
            for handler in tldr_ui.tldr.logger.handlers:
                if hasattr(handler, 'logs') and hasattr(handler, 'get_logs'):
                    # This is likely our StreamlitLogHandler
                    try:
                        handler_logs = handler.get_logs()
                        if handler_logs:
                            new_logs.extend(handler_logs)
                            handler.clear_logs()
                    except Exception as e:
                        print(f"Error reading handler logs: {e}")
        
        # Method 3: Add processing status logs
        if tldr_ui.processing:
            status_msg = f"INFO: {tldr_ui.status}"
            if status_msg not in st.session_state.output_lines:
                new_logs.append(status_msg)
                # Also print to console
                print(status_msg)
        
        # Method 4: Add completion status logs when processing finishes
        if hasattr(tldr_ui.tldr, 'total_input_tokens') and tldr_ui.tldr.total_input_tokens > 0:
            status_log = f"INFO: Processing complete - Input tokens: {tldr_ui.tldr.total_input_tokens}, Output tokens: {tldr_ui.tldr.total_output_tokens}, Total spend: ${tldr_ui.tldr.total_spend:.4f}"
            if status_log not in st.session_state.output_lines:
                new_logs.append(status_log)
                # Also print to console
                print(status_log)
        
        # Update session state with new logs
        if new_logs:
            st.session_state.output_lines.extend(new_logs)
            # Keep only last 100 lines to prevent memory issues
            st.session_state.output_lines = st.session_state.output_lines[-100:]
        
        # Display the logs in a scrollable container
        log_container = st.container(height=400)
        with log_container:
            if st.session_state.output_lines:
                # Display logs in chronological order (oldest first)
                for log in st.session_state.output_lines:
                    st.text(log)
            else:
                st.text("DEBUG: Logger status:")
                if hasattr(tldr_ui.tldr, 'logger'):
                    st.text(f"  - Logger exists: {type(tldr_ui.tldr.logger)}")
                    if hasattr(tldr_ui.tldr.logger, 'handlers'):
                        st.text(f"  - Handlers: {[type(h).__name__ for h in tldr_ui.tldr.logger.handlers]}")
                else:
                    st.text("  - No logger found on TldrEngine")

    # Status bar
    status_bar = st.container()
    with status_bar:
        st.markdown(
            f"""
            <div class='status-bar'>
                <span>Status: {tldr_ui.status}</span>
                <span>Author: Matt Jenior</span>
                <span>Version: {version} - 2025</span>
            </div>
            """,
            unsafe_allow_html=True,
        )



def start_ui():
    """Command-line entry point to run the tldr streamlit UI."""
    asyncio.run(run_tldr_streamlit())


if __name__ == "__main__":
    start_ui()
    