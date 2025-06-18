"""
TLDR Summary Generator - Streamlit Web Interface
"""
import os
import sys
import asyncio
import streamlit as st
from pathlib import Path
from typing import List, Union, BinaryIO
from streamlit.runtime.uploaded_file_manager import UploadedFile
#from streamlit.delta_generator import DeltaGenerator

# Add the parent directory to path to import tldr
sys.path.append(str(Path(__file__).parent))

# Import tldr components
from tldr.tldr import TldrEngine

# Page config
st.set_page_config(
    page_title="TLDR Summary Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

class TLDRApp:
    def __init__(self):
        self.uploaded_files = []
        self.summary = ""
        self.polished_summary = ""
        self.status = "Ready"
        self.token_count = 0
        self.total_spend = 0.0
        self.processing = False
        
        # Initialize session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'context' not in st.session_state:
            st.session_state.context = None
        if 'executive' not in st.session_state:
            st.session_state.executive = ""
        if 'polished' not in st.session_state:
            st.session_state.polished = ""
        if 'reference_summaries' not in st.session_state:
            st.session_state.reference_summaries = []

        


    def process_files(self, files: List[Union[UploadedFile, BinaryIO]]) -> List[dict]:
        """Process uploaded files and return file info"""
        file_info = []
        for file in files:
            file_size = len(file.getvalue())
            file_info.append({
                'path': file.name,
                'size': f"{file_size / 1024:.1f} KB",
            })
        return file_info




    async def run_async_function(self, async_func):
        """Run an async function with processing status updates"""
        try:
            self.status = "Processing..."
            self.processing = True
            await async_func()
            self.status = "Ready"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()
        finally:
            self.processing = False

    def document_uploader(self, header: str, key: str):
        """Document uploader"""
        st.subheader(header)
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key=key
        )
        return uploaded_files

async def main():
    # Initialize app
    tldr_ui = TLDRApp()
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Select options
        #st.subheader("Options")
        polish = st.checkbox("Polish Final Summary", value=True)
        web_search = st.checkbox("Enable Web Research", value=True)
        tone = st.selectbox("Polished summary tone", ["stylized", "formal"], index=0)
        context_size = st.selectbox("Context size", ["small", "medium", "large"], index=1)
        
        # Status
        st.divider()
        st.subheader("Status")
        st.text(f"Status: {tldr_ui.status}")
        st.text(f"Token Count: {tldr_ui.token_count}")
        st.text(f"Total Spend: ${tldr_ui.total_spend}")
    
    # Main content area
    st.title("📝 TLDR Summary Generator")
    st.markdown("Generate concise summaries from your documents with AI assistance")
    
    # Create two main columns directly
    col1, col2 = st.columns([1, 1], gap="medium")
    
    # Left column - Document input and controls
    with col1:

        # Query input
        st.subheader("Query")
        query = st.text_area(
            "What would you like to know from these documents?",
            placeholder="Enter your query here...",
            height=100,
            key="query_input"  # Add a key to reference this widget
        )
        
        # Initialize or update refined query
        if 'refined_query' not in st.session_state:
            st.session_state.refined_query = query
        
        # Upload files
        documents = tldr_ui.document_uploader("Target Documents", "document_uploader")
        context = tldr_ui.document_uploader("Additional Context", "context_uploader")
        
        # Process uploaded files
        if documents:
            if st.button("Process Input"):
                with st.spinner("Processing files..."):
                    st.session_state.documents = tldr_ui.process_files(documents)
                    if context:
                        st.session_state.context = tldr_ui.process_files(context)
                    else:
                        st.session_state.context = None

                    # Prepare arguments for TldrEngine
                    print([f['path'] for f in st.session_state.documents])
                    args = {
                        'input_files': [f['path'] for f in st.session_state.documents],
                        'query': query,
                        'context_files': [f['path'] for f in st.session_state.context] if st.session_state.context else None,
                        'polish': polish,
                        'web_search': web_search,
                        'context_size': context_size,
                        'tone': tone,
                        'pdf': False,
                        'verbose': False,
                    }
                    
                    # Initialize TldrEngine and load content
                    tldr_ui.tldr = TldrEngine(**args)
                    await tldr_ui.tldr.initialize_async_session()
                    
                    # Update the query with the refined version if available
                    if hasattr(tldr_ui.tldr, 'refined_query') and tldr_ui.tldr.refined_query:
                        st.session_state.refined_query = tldr_ui.tldr.refined_query
                        # Update the query widget
                        st.session_state.query_input = tldr_ui.tldr.refined_query
                    
                    # Generate component summaries
                    await tldr_ui.run_async_function(tldr_ui.tldr.summarize_resources)
                    st.session_state.reference_summaries = tldr_ui.tldr.reference_summaries

        # Display file list and document summary
        if st.session_state.documents:
            st.subheader("Uploaded Files")
            
            # Create two columns: one for file list, one for summary
            file_col, summary_col = st.columns([1, 2])
            
            with file_col:
                # Track selected document
                if 'selected_doc' not in st.session_state:
                    st.session_state.selected_doc = None
                
                # Display file list with selection
                for doc in st.session_state.documents:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            # Make file name clickable for selection
                            if st.button(doc['path'], key=f"select_{doc['path']}", use_container_width=True):
                                st.session_state.selected_doc = doc
                        with col2:
                            if st.button("🗑️", key=f"del_{doc['path']}"):
                                st.session_state.documents = [d for d in st.session_state.documents if d['path'] != doc['path']]
                                if st.session_state.selected_doc and st.session_state.selected_doc['path'] == doc['path']:
                                    st.session_state.selected_doc = None
                                st.rerun()
            
            with summary_col:
                st.subheader("Document Summary")
                if st.session_state.selected_doc:
                    # Display document summary in a scrollable container
                    summary = st.session_state.selected_doc.get('summary', 'No summary available')
                    st.markdown(
                        f'<div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; max-height: 300px; overflow-y: auto;">'
                        f'{summary}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Select a document to view its summary")
            if st.session_state.reference_summaries:
                st.subheader("Component Summaries")
                for summary in st.session_state.reference_summaries:
                    with st.container():
                        st.text(summary)

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Research", disabled=tldr_ui.processing):
                asyncio.run(tldr_ui.apply_research())
        with col2:
            if st.button("Integrate", disabled=not st.session_state.reference_summaries or tldr_ui.processing):
                asyncio.run(tldr_ui.integrate_summaries())
        with col3:
            if st.button("Polish", disabled=not st.session_state.executive or tldr_ui.processing):
                asyncio.run(tldr_ui.polish_response())

    with col2:
        st.subheader("Summaries")
        
        # Summary tabs
        tab1, tab2 = st.tabs(["Current", "Polished"])
        
        # Summary content
        with tab1:
            executive = st.text_area(
                "Summaries",
                value=st.session_state.executive,
                height=400,
                key="executive_output"
            )
        
        with tab2:
            polished = st.text_area(
                "Polished Summary",
                value=st.session_state.polished,
                height=400,
                key="polished_output"
            )
        
        # Action buttons for summary - using a single row
        st.write("")
        
        # Create a container for buttons
        btn_container = st.container()
        
        # Use f-strings for button layout with proper escaping
        disabled_attr = 'disabled' if not st.session_state.reference_summaries else ''
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
        if st.button("📋 Copy to Clipboard", 
                    key="copy_btn_hidden", 
                    disabled=not st.session_state.reference_summaries,
                    help="Copy summary to clipboard"):
            st.session_state.clipboard = st.session_state.reference_summaries
            st.toast("Copied to clipboard!")
        
        if st.button("💾 Save as PDF",
                   key="pdf_btn_hidden",
                   disabled=not st.session_state.polished,
                   help="Save summary as PDF"):
            await tldr.save_to_pdf(st.session_state.polished)
            st.toast("PDF saved!")
    
    # Status bar
    status_bar = st.container()
    with status_bar:
        st.markdown(
            f"""
            <div class='status-bar'>
                <span>Status: {tldr_ui.status}</span>
                <span>Tokens: {tldr_ui.token_count}</span>
                <span>v1.0.0</span>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    asyncio.run(main())
