"""
TLDR Summary Generator - Streamlit Web Interface
"""
version = "1.0.33"

import os
import sys
import math
import asyncio
import traceback
from pathlib import Path
from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from datetime import datetime, timedelta

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

    def reset_session_status(self):
        self.status = "Ready"
        self.processing = False

        st.session_state.status = "Ready"
        st.session_state.processing = False
        st.session_state.total_spend = self.round_up(self.tldr.total_spend)
        st.session_state.input_token_count = self.tldr.total_input_tokens
        st.session_state.output_token_count = self.tldr.total_output_tokens

    def round_up(self, num, to=0.01):
        """Round up to the nearest specified decimal place"""
        return round(num + to / 2, -int(math.log10(to)))

    async def session_load_all_content(
        self, input_files, context_files, context_size="medium"
    ):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await self.tldr.load_all_content(input_files, context_files, context_size)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

    async def session_refine_query(self, query, context_size="medium"):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await self.tldr.refine_user_query(query, context_size)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

    async def session_apply_research(self, context_size="medium"):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await self.tldr.apply_research(context_size)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

    async def session_summarize_resources(self, context_size="medium"):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await self.tldr.summarize_resources(context_size)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

    async def session_integrate_summaries(self, context_size="medium"):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            self.tldr.integrate_summaries(context_size)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

    async def session_polish_response(self, tone="stylized", context_size="medium"):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await self.tldr.polish_response(tone, context_size)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

    async def session_save_to_pdf(self, polished=True):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await self.tldr.save_to_pdf(polished)
        except Exception as e:
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()

        # Update session state
        self.reset_session_status()

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

    def document_uploader(self, header: str, key: str, description: str = None):
        """Document uploader with logging of uploaded files"""
        st.subheader(header)
        if description is not None:
            st.write(description)

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key=key,
        )
        
        # Log the uploaded files
        if uploaded_files and len(uploaded_files) > 0:
            file_names = ", ".join([f"'{file.name}'" for file in uploaded_files])
            if hasattr(self, 'tldr') and hasattr(self.tldr, 'logger'):
                self.tldr.logger.info(f"Uploaded {len(uploaded_files)} file(s) to {key}: {file_names}")
        
        return uploaded_files


async def main():

    # Initialize session state
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "context_files" not in st.session_state:
        st.session_state.context_files = None
    if "added_context" not in st.session_state:
        st.session_state.added_context = None
    if "executive" not in st.session_state:
        st.session_state.executive = None
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "polished" not in st.session_state:
        st.session_state.polished = None
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
    tldr_ui = st.session_state.tldr_ui

    # Handle refined query update
    if "refined_query" in st.session_state:
        st.session_state.user_query = st.session_state.refined_query
        del st.session_state.refined_query

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
        st.markdown(
            """> **Input**
- **Target Documents:** The primary files you want to summarize.  
- **Additional Context:** Optional supplementary documents to provide more context for the summarization.
- **Focused Query:** The specific question or topic you want to focus on."""
        )

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
        if documents is not None:
            st.markdown('<div class="process-button">', unsafe_allow_html=True)
            process_clicked = st.button("Upload Documents")
            st.markdown("</div>", unsafe_allow_html=True)

            if process_clicked:
                with st.spinner("Processing documents..."):
                    st.session_state.documents = tldr_ui.process_files(documents)
                    input_files = [f["source"] for f in st.session_state.documents]
                    if context:
                        st.session_state.context_files = tldr_ui.process_files(context)
                        context_files = [
                            f["source"] for f in st.session_state.context_files
                        ]
                    else:
                        st.session_state.context_files = None
                        context_files = None

                    # Collect all content
                    await tldr_ui.session_load_all_content(
                        input_files=input_files,
                        context_files=context_files,
                        context_size=context_size,
                    )
                    # And update the session state
                    for doc in st.session_state.documents:
                        doc["content"] = tldr_ui.tldr.content[doc["source"]]

                    if st.session_state.context_files is not None:
                        st.session_state.added_context = tldr_ui.tldr.added_context

        # Query input
        st.subheader("Focused Query")
        st.text_area(
            "What would you like to know from these documents?",
            height=70,
            key="user_query",
        )

        # Query refine button, display refined text once returned
        if st.button(
            "Refine Query",
            disabled=not st.session_state.user_query or tldr_ui.processing,
        ):
            with st.spinner("Refining query..."):
                await tldr_ui.session_refine_query(st.session_state.user_query)
                st.session_state.refined_query = tldr_ui.tldr.query
                st.rerun()

        # Display file listdoc
        if st.session_state.documents is not None:
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
        if st.session_state.selected_doc:
            st.subheader(f"Content of {st.session_state.selected_doc['source']}")
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
            else:
                st.info("Please upload and process the documents first to view content.")

    # Right column - Summaries and actions
    with col2:
        # Explanations for actions
        st.markdown(
            """> **Actions**
- **Generate Reference Summaries**: Summarizes the initial set of documents to create a baseline.
- **Refine Query**: Refines your initial query for better search results.
- **Apply Research**: Applies external research to the compiled document summaries.
- **Integrate Summaries**: Integrates all summaries with research and additional context into a cohesive whole.
- **Polish**: Polishes the final summary for tone and style.
"""
        )
        # Generate initial summaries
        if st.button("Generate Reference Summaries", disabled=st.session_state.documents is None):
            with st.spinner("Summarizing documents..."):

                # Generate document summaries
                await tldr_ui.session_summarize_resources(context_size=context_size)
                # And update the session state
                st.session_state.summarized = True
                for doc in st.session_state.documents:
                    doc["summary"] = tldr_ui.tldr.content[doc["source"]]["summary"]

        st.subheader("Document Summaries")
        # Display selected document content
        if st.session_state.selected_doc and st.session_state.summarized is True:
            st.subheader(f"Summary of {st.session_state.selected_doc['source']}")
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
                or tldr_ui.processing is True,
            ):
                with st.spinner("Researching knowledge gaps..."):
                    await tldr_ui.session_apply_research(context_size=context_size)

        # Synthesis
        with action_col2:
            if st.button(
                "Synthesis",
                disabled=st.session_state.documents is None
                or tldr_ui.processing is True,
            ):
                with st.spinner(
                    "Synthesizing summaries, research, and new added context..."
                ):
                    await tldr_ui.session_integrate_summaries(context_size=context_size)
    
        # Polish
        with action_col3:
            if st.button(
                "Polish",
                disabled=st.session_state.executive is None
                or tldr_ui.processing is True,
            ):
                with st.spinner("Polishing finalized summary..."):
                    await tldr_ui.session_polish_response(
                        tone=tone, context_size=context_size
                    )

        st.subheader("TLDR Text")
        # Summary tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Added Context", "Research Results", "Executive Summary", "Polished Summary"]
        )

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
        
        # Added Context Tab
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✏️ Edit", key="edit_added_context"):
                    st.session_state.edit_mode['added_context'] = True
            with col2:
                if st.session_state.edit_mode['added_context'] and st.button("💾 Save", key="save_added_context"):
                    st.session_state.edit_mode['added_context'] = False
                    if hasattr(tldr_ui, 'tldr'):
                        tldr_ui.tldr.added_context = st.session_state.edit_buffers['added_context']
            
            st.text_area(
                "Additional context provided during executive summary generation",
                height=400,
                value=st.session_state.edit_buffers['added_context'],
                key="added_context_area",
                disabled=not st.session_state.edit_mode['added_context'],
                on_change=lambda: st.session_state.edit_buffers.update({
                    'added_context': st.session_state.added_context_area
                })
            )
        
        # Research Results Tab
        with tab2:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✏️ Edit", key="edit_research_context"):
                    st.session_state.edit_mode['research_context'] = True
            with col2:
                if st.session_state.edit_mode['research_context'] and st.button("💾 Save", key="save_research_context"):
                    st.session_state.edit_mode['research_context'] = False
                    if hasattr(tldr_ui, 'tldr'):
                        tldr_ui.tldr.research_context = st.session_state.edit_buffers['research_context']
            
            st.text_area(
                "Research results from knowledge gaps identified in document summaries",
                height=400,
                value=st.session_state.edit_buffers['research_context'],
                key="research_results_area",
                disabled=not st.session_state.edit_mode['research_context'],
                on_change=lambda: st.session_state.edit_buffers.update({
                    'research_context': st.session_state.research_results_area
                })
            )
        
        # Executive Summary Tab
        with tab3:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✏️ Edit", key="edit_exec_summary"):
                    st.session_state.edit_mode['executive_summary'] = True
            with col2:
                if st.session_state.edit_mode['executive_summary'] and st.button("💾 Save", key="save_exec_summary"):
                    st.session_state.edit_mode['executive_summary'] = False
                    if hasattr(tldr_ui, 'tldr'):
                        tldr_ui.tldr.executive_summary = st.session_state.edit_buffers['executive_summary']
            
            st.text_area(
                "Executive summary of combined document summaries and research results",
                height=400,
                value=st.session_state.edit_buffers['executive_summary'],
                key="executive_area",
                disabled=not st.session_state.edit_mode['executive_summary'],
                on_change=lambda: st.session_state.edit_buffers.update({
                    'executive_summary': st.session_state.executive_area
                })
            )
        
        # Polished Summary Tab
        with tab4:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✏️ Edit", key="edit_polished"):
                    st.session_state.edit_mode['polished_summary'] = True
            with col2:
                if st.session_state.edit_mode['polished_summary'] and st.button("💾 Save", key="save_polished"):
                    st.session_state.edit_mode['polished_summary'] = False
                    if hasattr(tldr_ui, 'tldr'):
                        tldr_ui.tldr.polished_summary = st.session_state.edit_buffers['polished_summary']
            
            st.text_area(
                "Polished executive summary with improved formatting and tone",
                height=400,
                value=st.session_state.edit_buffers['polished_summary'],
                key="polished_area",
                disabled=not st.session_state.edit_mode['polished_summary'],
                on_change=lambda: st.session_state.edit_buffers.update({
                    'polished_summary': st.session_state.polished_area
                })
            )
        st.write("")

        st.subheader("Output")
        # Create three columns for buttons
        output_col1, output_col2 = st.columns(2)

        with output_col1:
            # Add the actual buttons (invisible, just for the click handlers)
            if st.button(
                "📋 Copy to Clipboard",
                key="copy_btn_hidden",
                disabled=not st.session_state.selected_doc and st.session_state.summarized is False,
                help="Copy text to clipboard",
            ):
                st.session_state.clipboard = st.session_state.selected_doc["summary"]
                st.toast("Copied to clipboard!")

        with output_col2:
            if st.button(
                "💾 Save as PDF",
                key="pdf_btn_hidden",
                disabled=not st.session_state.polished or not st.session_state.executive,
                help="Save summary as PDF",
            ):
                with st.spinner("Generating PDF..."):
                    await tldr_ui.tldr.save_to_pdf(st.session_state.polished)
                st.toast("PDF saved!")

    # Status and Output section
    with st.expander("🔍 Processing Logs", expanded=False):
        # Initialize session state for logs if needed
        if 'output_lines' not in st.session_state:
            st.session_state.output_lines = []
        
        # Get the StreamlitLogHandler if it exists
        streamlit_handler = None
        if hasattr(tldr_ui.tldr, 'logger') and hasattr(tldr_ui.tldr.logger, 'streamlit_handler'):
            streamlit_handler = tldr_ui.tldr.logger.streamlit_handler
        
        # Clear logs button
        if st.button("Clear Logs"):
            if streamlit_handler is not None:
                streamlit_handler.clear_logs()
            st.session_state.output_lines = []
        
        # Update logs from the handler
        if streamlit_handler is not None:
            logs = streamlit_handler.get_logs()
            if logs:
                st.session_state.output_lines = logs[-100:]  # Keep last 100 lines
        
        # Display the logs in a scrollable container
        log_container = st.container(height=400)
        with log_container:
            if st.session_state.output_lines:
                # Display logs in reverse order (newest first)
                for log in reversed(st.session_state.output_lines):
                    st.text(log)
                
                # Auto-scroll to bottom
                st.markdown(
                    """
                    <script>
                        const logs = document.querySelectorAll('.stText');
                        if (logs.length > 0) {
                            logs[logs.length - 1].scrollIntoView();
                        }
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No logs available. Processing will generate logs here.")

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


if __name__ == "__main__":
    asyncio.run(main())
