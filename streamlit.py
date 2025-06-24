"""
TLDR Summary Generator - Streamlit Web Interface
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from typing import List, Union, BinaryIO

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

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
        st.session_state.total_spend = round(self.tldr.total_spend, 2) + 0.01
        st.session_state.input_token_count = self.tldr.total_input_tokens
        st.session_state.output_token_count = self.tldr.total_output_tokens

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

    def document_uploader(self, header: str, key: str):
        """Document uploader"""
        st.subheader(header)

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key=key,
        )
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
        tone = st.selectbox("Polished summary tone", ["stylized", "formal"], index=0)
        context_size = st.selectbox(
            "Context size", ["small", "medium", "large"], index=1
        )
        # Status
        st.divider()
        st.subheader("Status")
        st.text(f"Status: {tldr_ui.status}")
        st.text(f"Input Token Count: {st.session_state.input_token_count}")
        st.text(f"Output Token Count: {st.session_state.output_token_count}")
        st.text(f"Total Spend: ~${st.session_state.total_spend}")

    # Main content area
    st.title("📝 TLDR Summary Generator")
    st.markdown("Generate concise summaries from your documents with AI assistance")

    # Create two main columns directly
    col1, col2 = st.columns([1, 1], gap="medium")

    # Left column - Document input and controls
    with col1:

        # Upload files
        documents = tldr_ui.document_uploader("Target Documents", "document_uploader")
        context = tldr_ui.document_uploader("Additional Context", "context_uploader")

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
        st.subheader("Query")
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

            # Generate initial summaries
            if st.button("Generate Reference Summaries"):
                with st.spinner("Summarizing documents..."):

                    # Generate document summaries
                    await tldr_ui.session_summarize_resources(context_size=context_size)
                    # And update the session state
                    st.session_state.summarized = True
                    for doc in st.session_state.documents:
                        doc["summary"] = tldr_ui.tldr.content[doc["source"]]["summary"]

    # Right column - Summaries and actions
    with col2:
        # Display selected document content
        if st.session_state.selected_doc:
            st.subheader(f"Content of {st.session_state.selected_doc['source']}")
            # Display content in a scrollable text area
            st.text_area(
                "Document Content",
                value=st.session_state.selected_doc["content"],
                height=300,
                key=f"content_{st.session_state.selected_doc['source']}",
                disabled=True,
            )

            st.subheader("Document Summary")
            if st.session_state.summarized is True:
                st.info("Select a document to view its summary")
                # Display document summary in a scrollable container
                summary = st.session_state.selected_doc.get(
                    "summary", "No summary available"
                )
                st.markdown(
                    f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                    f"{summary}"
                    "</div>",
                    unsafe_allow_html=True,
                )

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
                with st.spinner("Researching gaps..."):
                    await tldr_ui.session_apply_research(context_size=context_size)
                # Update session
                st.session_state.research_results = tldr_ui.tldr.research_results

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
                # Update session
                st.session_state.executive = tldr_ui.tldr.executive_summary

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
                # Update session
                st.session_state.polished = tldr_ui.tldr.polished_summary

        st.subheader("Summaries")

        # Summary tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Added Context", "Research", "Executive", "Polished"]
        )

        # Summary content
        with tab1:
            st.text_area(
                "Added Context",
                height=400,
                key="added_context",
            )
        with tab2:
            st.text_area(
                "Research Results",
                height=400,
                key="research_results",
            )
        with tab3:
            st.text_area(
                "Executive Summary",
                height=400,
                key="executive",
            )
        with tab4:
            st.text_area(
                "Polished Summary",
                height=400,
                key="polished",
            )
        # Action buttons for summary - using a single row
        st.write("")

        # Create a container for buttons
        btn_container = st.container()

        # Use f-strings for button layout with proper escaping
        disabled_attr = "disabled" if st.session_state.summarized is False else ""
        btn_html = f"""
        <style>
            .button-container {{
                display: flex;
                gap: 10px;
                width: 100%;
            }}
            .button-container button {{
                flex: 1;
            }}
        </style>
        <div class="button-container">
            <button class="stButton" onclick="copyToClipboard()" {disabled_attr}>📋 Copy to Clipboard</button>
            <button class="stButton" onclick="saveAsPDF()" {disabled_attr}>💾 Save as PDF</button>
        </div>
        <script>
            function copyToClipboard() {{
                // This will be handled by Streamlit's button
            }}
            function saveAsPDF() {{
                // This will be handled by Streamlit's button
            }}
        </script>
        """

        # Add the HTML to the container
        btn_container.markdown(btn_html, unsafe_allow_html=True)

        # Add the actual buttons (invisible, just for the click handlers)
        if st.button(
            "📋 Copy to Clipboard",
            key="copy_btn_hidden",
            disabled=st.session_state.summarized is False,
            help="Copy summary to clipboard",
        ):
            st.session_state.clipboard = st.session_state.reference_summaries
            st.toast("Copied to clipboard!")

        if st.button(
            "💾 Save as PDF",
            key="pdf_btn_hidden",
            disabled=not st.session_state.polished or not st.session_state.executive,
            help="Save summary as PDF",
        ):
            await tldr_ui.tldr.save_to_pdf(st.session_state.polished)
            st.toast("PDF saved!")

    # Status bar
    status_bar = st.container()
    with status_bar:
        st.markdown(
            f"""
            <div class='status-bar'>
                <span>Status: {tldr_ui.status}</span>
                <span>Author: Matt Jenior</span>
                <span>Version: 1.0.0 - 2025</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
