"""
TLDR Summary Generator - Streamlit Web Interface
"""

version = "1.2.0"

import os
import math
import asyncio
import traceback
from typing import List

import streamlit as st

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

    def start_processing_status(self, message: str):
        """Helper method to update processing status"""
        self.tldr.logger.info(message)
        self.status = "Processing..."
        self.processing = True
        st.session_state.status = self.status
        st.session_state.processing = self.processing

    def end_processing_status(self):
        """Reset session status and update session state"""
        self.tldr.logger.info("Completed successfully")

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
        self.tldr.logger.error(f"Error: {error_message}")
        self.tldr.logger.error(f"Error details: {error_details}")
        st.error(
            f"An error occurred: {error_message}\n\nError details:\n{error_details}"
        )
        st.stop()

    @staticmethod
    def round_up(num, to=0.01):
        """Round up to the nearest specified decimal"""
        return round(num + to / 2, -int(math.log10(to)))

    async def execute_session_process(
        self, log_message: str, function: callable, **args
    ):
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
        if uploaded_files is not None and len(uploaded_files) > 0:
            file_names = ", ".join([f"'{file.name}'" for file in uploaded_files])
            self.tldr.logger.info(
                f"Uploaded {len(uploaded_files)} file(s) to {key}: {file_names}"
            )

        return uploaded_files


async def run_tldr_streamlit():

    # Initialize console logging from the start
    print(f"TLDR v{version}: Starting Streamlit application")
    print(
        "TLDR: Console logging is enabled - all operations will be logged to terminal"
    )

    # Initialize session state
    if "api_provider" not in st.session_state:
        st.session_state.api_provider = "openai"
    st.session_state.processing = False
    st.session_state.status = "Ready"
    st.session_state.input_token_count = 0
    st.session_state.output_token_count = 0
    st.session_state.total_spend = 0.0
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

    # Initialize TLDR classes if first run
    if "tldr_ui" not in st.session_state:
        st.session_state.tldr_ui = TldrUI()
        st.session_state.current_platform = "OpenAI"  # Default platform
        st.session_state.tldr_ui.tldr = TldrEngine(platform=st.session_state.current_platform.lower())

        # Log initialization success
        print("TLDR: TldrEngine initialized successfully")
        print("TLDR: StreamlitLogHandler is ready for log capture")

    # Finish initialization
    tldr_ui = st.session_state.tldr_ui
    tldr_ui.tldr.logger.info("TLDR system initialized and ready for use")

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
        st.subheader("Options", divider=True)
        
        # Initialize platform in session state if not exists
        if "current_platform" not in st.session_state:
            st.session_state.current_platform = None
            
        # Platform selection with callback to reinitialize TldrEngine
        def on_platform_change():
            if st.session_state.current_platform != st.session_state.platform_selector:
                st.session_state.current_platform = st.session_state.platform_selector
                st.session_state.tldr_ui.tldr = TldrEngine(platform=st.session_state.platform_selector.lower())
                st.rerun()
        platform = st.radio(
            "API Provider",
            ["OpenAI", "Google"],
            index=0 if st.session_state.current_platform is None else ["OpenAI", "Google"].index(st.session_state.current_platform),
            help="The API provider to use (Changing reinitializes the client).",
            key="platform_selector",
            on_change=on_platform_change
        )
        # Other options
        tone = st.selectbox(
            "Polished summary tone",
            ["stylized", "formal"],
            index=0,
            help="The tone of the polished summary (Stylized follows writing style of certain prominent authors).",
        )
        context_size = st.selectbox(
            "Context window",
            ["small", "medium", "large"],
            index=1,
            help="The context window size and research effort.",
        )
        # Status
        st.divider()
        st.subheader("Status", divider=True)
        st.text(f"System: {st.session_state.status}")
        st.text(f"Input Token Count: {st.session_state.input_token_count}")
        st.text(f"Output Token Count: {st.session_state.output_token_count}")
        st.text(f"Approx. Total Spend: ${st.session_state.total_spend}")

    # Main content area
    st.title(f"📝 Too Long; Didn't Read (TLDR) - v{version}")
    st.markdown(
        "Generate concise summaries, research knowledge gaps, and synthesize information from your documents with AI assistance"
    )

    # Create two main columns directly
    col1, col2 = st.columns([1, 1], gap="medium")

    # Left column - Document input and controls
    with col1:
        st.subheader(
            "Document Upload",
            help="Upload documents for summary and extra context.",
            divider=True,
        )
        # Upload files
        documents = tldr_ui.document_uploader(
            "Target Documents",
            "document_uploader",
            "Selected documents to focus on for summaries and research",
        )
        context = tldr_ui.document_uploader(
            "Additional Context",
            "context_uploader",
            "Documents to provide supplementary context",
        )

        # Process uploaded files
        if documents is not None or context is not None:
            st.markdown('<div class="process-button">', unsafe_allow_html=True)
            process_clicked = st.button(
                "Process References",
                disabled=st.session_state.processing is True,
                help="Process the documents you want to summarize and research.",
            )
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
                        search=False,
                    )
                    # And update the session state
                    for doc in st.session_state.documents:
                        doc["content"] = tldr_ui.tldr.content[doc["source"]]
                    if context_files is not None:
                        st.session_state.added_context = tldr_ui.tldr.added_context

        # Query input
        st.subheader("Focused Query", divider=True)
        st.text_area(
            "What would you like to know from these documents?",
            height=70,
            key="user_query",
            disabled=st.session_state.processing is True,
        )

        query_col1, query_col2 = st.columns(2)

        # Query refine button, display refined text once returned
        with query_col1:
            if st.button(
                "Refine Query",
                disabled=st.session_state.user_query is None
                or st.session_state.processing is True,
                help="Refine the user query to be more specific and comprehensive.",
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
                "Web Search for Context",
                disabled=st.session_state.user_query is None
                or st.session_state.processing is True,
                help="Search web for additional context based on current user query.",
            ):
                with st.spinner("Searching web for improved context..."):
                    await tldr_ui.execute_session_process(
                        log_message=f"Searching for context: {st.session_state.user_query}",
                        function=tldr_ui.tldr.initial_context_search,
                        context_size=context_size,
                    )
                    st.session_state.initial_context = tldr_ui.tldr.added_context
                    st.rerun()

    # Right column - Summaries and actions
    with col2:
        # Display file listdoc
        if st.session_state.documents is not None:
            st.subheader("Input References", divider=True)
            st.subheader("Manage Files")

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
                            help="Select this document to view its content.",
                            disabled=st.session_state.processing is True,
                        ):
                            st.session_state.selected_doc = doc
                    with del_btn_col:
                        if st.button(
                            "🗑️",
                            key=f"del_{doc['source']}",
                            disabled=st.session_state.processing is True,
                            help="Delete this document.",
                        ):
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
                st.text(
                    "Select a document from the list of uploaded documents to view its content."
                )
            elif st.session_state.selected_doc is not None:
                st.subheader("Document Content")
                # Display content in a scrollable text area
                if (
                    "content" in st.session_state.selected_doc
                    and st.session_state.selected_doc["content"]
                ):
                    content = st.session_state.selected_doc["content"]
                    if isinstance(content, str):
                        content_html = content.replace("\n", "<br>")
                        st.markdown(
                            f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                            f"{content_html}"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif isinstance(content, dict):
                        # Get actual text content
                        text_content = content.get("content", None)
                        if text_content:
                            content_html = text_content.replace("\n", "<br>")
                            st.markdown(
                                f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                                f"{content_html}"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # Show the structure for debugging
                            st.write("Content structure:")
                            st.json(content)
                    else:
                        st.warning("Document content is not in a readable format.")
        st.write("")

        # Generate initial summaries
        if st.button(
            "Generate Summaries",
            disabled=st.session_state.documents is None
            or st.session_state.processing is True,
            help="Summarizes the initial set of documents to create a baseline.",
        ):
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
                st.rerun()

        # Display selected document content
        if st.session_state.summarized is True:
            st.subheader(
                "Document Summary", help="View the summaries of the uploaded documents."
            )
            if st.session_state.selected_doc is None:
                st.text(
                    "Select a document from the list of uploaded documents to view its summary."
                )
            else:
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
            st.info("Generate reference summaries first to access other features.")

        # Create three columns for buttons
        if st.session_state.summarized is True:
            st.subheader(
                "Summary Tools",
                help="Generate summaries, research, and integrate TLDR text.",
                divider=True,
            )

            action_col1, action_col2, action_col3 = st.columns(3)

            # Research
            with action_col1:
                if st.button(
                    "Web Research",
                    disabled=st.session_state.summarized is False
                    or st.session_state.processing is True,
                    help="Apply external research to the compiled document summaries",
                ):
                    with st.spinner("Researching knowledge gaps (please wait)..."):
                        await tldr_ui.execute_session_process(
                            log_message="Researching knowledge gaps...",
                            function=tldr_ui.tldr.apply_research,
                            context_size=context_size,
                        )
                        st.session_state.research_results = (
                            tldr_ui.tldr.research_results
                        )
                        st.session_state.added_context = tldr_ui.tldr.added_context
                        st.rerun()

            # Synthesis
            with action_col2:
                if st.button(
                    "Integrate Summaries",
                    disabled=st.session_state.summarized is False
                    or st.session_state.processing is True,
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
                        st.session_state.executive_summary = (
                            tldr_ui.tldr.executive_summary
                        )
                        st.rerun()

            # Polish
            with action_col3:
                if st.button(
                    "Polish Summary",
                    disabled=st.session_state.executive_summary is None
                    or st.session_state.processing is True,
                    help="Polish the finalized summary",
                ):
                    with st.spinner("Polishing finalized summary..."):
                        await tldr_ui.execute_session_process(
                            log_message="Polishing finalized summary...",
                            function=tldr_ui.tldr.polish_response,
                            tone=tone,
                            context_size=context_size,
                        )
                        st.session_state.polished_summary = (
                            tldr_ui.tldr.polished_summary
                        )
                        st.rerun()

        if (
            st.session_state.added_context is not None
            or st.session_state.research_results is not None
            or st.session_state.executive_summary is not None
        ):

            # Summary tabs
            tabs = st.tabs(
                [
                    "Added Context",
                    "Research Results",
                    "Executive Summary",
                    "Polished Summary",
                ]
            )
            tab_names = [
                "added_context",
                "research_results",
                "executive_summary",
                "polished_summary",
            ]

            # Update active tab in session state
            for i, tab in enumerate(tabs):
                with tab:
                    # This will run when the tab is active
                    st.session_state.active_tab = tab_names[i]

            # Summary content
            # Initialize edit mode state if not exists
            if "edit_mode" not in st.session_state:
                st.session_state.edit_mode = {
                    "added_context": False,
                    "research_results": False,
                    "executive_summary": False,
                    "polished_summary": False,
                }

            # Initialize edit buffers if not exists
            if "edit_buffers" not in st.session_state:
                st.session_state.edit_buffers = {
                    "added_context": "",
                    "research_results": "",
                    "executive_summary": "",
                    "polished_summary": "",
                }

            # Update edit buffers from TldrEngine if not in edit mode
            if hasattr(tldr_ui, "tldr"):
                if not st.session_state.edit_mode["added_context"] and hasattr(
                    tldr_ui.tldr, "added_context"
                ):
                    st.session_state.edit_buffers["added_context"] = (
                        tldr_ui.tldr.added_context or ""
                    )
                if not st.session_state.edit_mode["research_results"] and hasattr(
                    tldr_ui.tldr, "research_results"
                ):
                    st.session_state.edit_buffers["research_results"] = (
                        tldr_ui.tldr.research_results or ""
                    )
                if not st.session_state.edit_mode["executive_summary"] and hasattr(
                    tldr_ui.tldr, "executive_summary"
                ):
                    st.session_state.edit_buffers["executive_summary"] = (
                        tldr_ui.tldr.executive_summary or ""
                    )
                if not st.session_state.edit_mode["polished_summary"] and hasattr(
                    tldr_ui.tldr, "polished_summary"
                ):
                    st.session_state.edit_buffers["polished_summary"] = (
                        tldr_ui.tldr.polished_summary or ""
                    )

            # Tab contents
            tab_contents = [
                {
                    "title": "Additional context provided during research",
                    "key": "added_context",
                    "help_text": "Edit additional context",
                },
                {
                    "title": "Research results from external sources",
                    "key": "research_results",
                    "help_text": "Edit research results",
                },
                {
                    "title": "Executive summary of combined document summaries and research results",
                    "key": "executive_summary",
                    "help_text": "Edit executive summary",
                },
                {
                    "title": "Polished executive summary with improved formatting and tone",
                    "key": "polished_summary",
                    "help_text": "Edit polished summary",
                },
            ]

            # Create tab contents
            for i, tab in enumerate(tabs):
                with tab:
                    tab_info = tab_contents[i]
                    st.text_area(
                        tab_info["title"],
                        value=st.session_state.edit_buffers[tab_info["key"]],
                        key=f"{tab_info['key']}_area",
                        disabled=not st.session_state.edit_mode[tab_info["key"]],
                        on_change=lambda k=tab_info[
                            "key"
                        ]: st.session_state.edit_buffers.update(
                            {k: st.session_state[f"{k}_area"]}
                        ),
                    )
                    tab_col1, tab_col2 = st.columns([1, 1])
                    with tab_col1:
                        if st.button(f"✏️ Edit", key=f"edit_{tab_info['key']}"):
                            st.session_state.edit_mode[tab_info["key"]] = True
                    with tab_col2:
                        if st.session_state.edit_mode[tab_info["key"]] and st.button(
                            "💾 Save", key=f"save_{tab_info['key']}"
                        ):
                            st.session_state.edit_mode[tab_info["key"]] = False
                            setattr(
                                tldr_ui.tldr,
                                tab_info["key"],
                                st.session_state.edit_buffers[tab_info["key"]],
                            )
            st.write("")

            st.subheader(
                "Save Results",
                help="Save summaries, research, and integrate TLDR text.",
                divider=True,
            )
            # Create three columns for buttons
            output_col1, output_col2, output_col3 = st.columns(3)

            with output_col1:
                # Determine which tab is active and get its content
                active_tab = st.session_state.get("active_tab", "executive_summary")
                tab_content = st.session_state.edit_buffers.get(active_tab, None)

                if st.button(
                    "📋 Copy to Clipboard",
                    key="copy_btn_hidden",
                    disabled=tab_content is None,
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

        elif st.session_state.summarized is True:
            st.text("Use provided tools to view TLDR-generated text.")

    # Status and Output section
    with st.expander("🔍 Processing Logs", expanded=False):
        # Initialize session state for logs if needed
        if "output_lines" not in st.session_state:
            st.session_state.output_lines = []

        # Clear logs button
        if st.button("Clear Logs"):
            st.session_state.output_lines = []

        # Try to get logs from different possible sources
        new_logs = []

        # Method 1: Check for StreamlitLogHandler if it exists
        if (
            hasattr(tldr_ui.tldr, "streamlit_handler")
            and tldr_ui.tldr.streamlit_handler is not None
        ):
            try:
                logs = tldr_ui.tldr.streamlit_handler.get_logs()
                if logs:
                    new_logs.extend(logs)
                    # Clear the handler logs after reading to prevent duplication
                    tldr_ui.tldr.streamlit_handler.clear_logs()
            except Exception as e:
                tldr_ui.tldr.logger.error(f"Error reading streamlit handler logs: {e}")

        # Method 2: Check for any log records from the logger directly
        if hasattr(tldr_ui.tldr, "logger") and hasattr(tldr_ui.tldr.logger, "handlers"):
            for handler in tldr_ui.tldr.logger.handlers:
                if hasattr(handler, "logs") and hasattr(handler, "get_logs"):
                    # This is likely our StreamlitLogHandler
                    try:
                        handler_logs = handler.get_logs()
                        if handler_logs:
                            new_logs.extend(handler_logs)
                            handler.clear_logs()
                    except Exception as e:
                        tldr_ui.tldr.logger.error(f"Error reading handler logs: {e}")

        # Method 3: Add processing status logs
        if st.session_state.processing:
            status_msg = f"INFO: {tldr_ui.status}"
            if status_msg not in st.session_state.output_lines:
                new_logs.append(status_msg)
                tldr_ui.tldr.logger.info(status_msg)

        # Method 4: Add completion status logs when processing finishes
        if (
            hasattr(tldr_ui.tldr, "total_input_tokens")
            and tldr_ui.tldr.total_input_tokens > 0
        ):
            status_log = f"INFO: Processing complete - Input tokens: {tldr_ui.tldr.total_input_tokens}, Output tokens: {tldr_ui.tldr.total_output_tokens}, Total spend: ${tldr_ui.tldr.total_spend:.4f}"
            if status_log not in st.session_state.output_lines:
                new_logs.append(status_log)
                tldr_ui.tldr.logger.info(status_log)

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
                if hasattr(tldr_ui.tldr, "logger"):
                    st.text(f"  - Logger exists: {type(tldr_ui.tldr.logger)}")
                    if hasattr(tldr_ui.tldr.logger, "handlers"):
                        st.text(
                            f"  - Handlers: {[type(h).__name__ for h in tldr_ui.tldr.logger.handlers]}"
                        )
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


def start_tldr_ui():
    """Command-line entry point to run the tldr streamlit UI."""
    from pathlib import Path

    # Get the path to the current file
    streamlit_file = Path(__file__).resolve()

    # Run streamlit with the current file
    import subprocess

    subprocess.run(["streamlit", "run", str(streamlit_file)], check=True)


if __name__ == "__main__":
    start_tldr_ui()
