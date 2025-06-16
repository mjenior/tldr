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
        self.processing = False
        
        # Initialize session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'summary' not in st.session_state:
            st.session_state.summary = ""
        if 'polished' not in st.session_state:
            st.session_state.polished = ""

        


    def process_files(self, files: List[Union[UploadedFile, BinaryIO]]) -> List[dict]:
        """Process uploaded files and return file info"""
        file_info = []
        for file in files:
            file_path = os.path.join("uploads", file.name)
            # Get file size before writing
            file_size = len(file.getvalue())
            # Write file content
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_info.append({
                'name': file.name,
                'size': f"{file_size / 1024:.1f} KB",
                'path': file_path
            })
        return file_info

    async def generate_summary(self, query: str, context_files: List[str] = None, 
                            polish: bool = False, web_search: bool = False):
        """Generate summary using TldrEngine
        
        Args:
            query: The query to generate a summary for
            context_files: Optional list of context file paths
            polish: Whether to polish the summary
            web_search: Whether to enable web research
        """
        try:
            self.status = "Processing..."
            self.processing = True
            
            # Prepare arguments for TldrEngine
            args = {
                'input_files': [f['path'] for f in st.session_state.documents],
                'query': query,
                'context_files': context_files or [],
                'polish': polish,
                'web_search': web_search
            }
            
            # Initialize TldrEngine
            tldr = TldrEngine(**args)
            
            # Generate embeddings if query exists
            if tldr.query is not None and tldr.query.strip():
                tldr.encode_text_to_vector_store()
                await tldr.refine_user_query()
                if tldr.query.strip():
                    tldr.search_embedded_context(query=tldr.query)
            
            # Process context if provided
            if hasattr(tldr, 'raw_context') and tldr.raw_context is not None:
                await tldr.format_context_references()
            
            # Generate summary
            await tldr.summarize_resources()
            await tldr.integrate_summaries()
            
            # Apply research if needed
            if web_search and hasattr(tldr, 'web_search') and tldr.web_search:
                await tldr.apply_research()
            
            # Polish if requested
            if polish:
                await tldr.polish_response()
                self.polished_summary = getattr(tldr, 'polished_summary', '')
                st.session_state.polished = self.polished_summary or "No polished summary available."
            else:
                self.summary = getattr(tldr, 'executive_summary', '')
                st.session_state.summary = self.summary or "No summary available."
            
            # Update status
            if hasattr(tldr, 'format_spending'):
                tldr.format_spending()
            if hasattr(tldr, 'generate_token_report'):
                tldr.generate_token_report()
            self.token_count = getattr(tldr, 'total_tokens_used', 0)
            self.status = "Ready"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.status = "Error during processing"
            st.error(f"An error occurred: {str(e)}\n\nError details:\n{error_details}")
            st.stop()
        finally:
            self.processing = False

def main():
    # Initialize app
    app = TLDRApp()
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Context size
        context_size = st.selectbox(
            "Context Size",
            ["small", "medium", "large"],
            index=1
        )
        
        # Tone
        tone = st.selectbox(
            "Tone",
            ["stylized", "formal"],
            index=0
        )
        
        # Additional options
        st.subheader("Options")
        recursive_search = st.checkbox("Recursive Search", value=False)
        web_search = st.checkbox("Enable Web Research", value=False)
        polish = st.checkbox("Polish Final Summary", value=True)
        
        # Status
        st.divider()
        st.subheader("Status")
        st.text(f"Status: {app.status}")
        st.text(f"Token Count: {app.token_count}")
    
    # Main content area
    st.title("📝 TLDR Summary Generator")
    st.markdown("Generate concise summaries from your documents with AI assistance")
    
    # Create two main columns directly
    col1, col2 = st.columns([1, 1], gap="medium")
    
    # Left column - Document input and controls
    with col1:
        st.subheader("Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        # Context uploader
        context_files = st.file_uploader(
            "Upload context documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="context_uploader"
        )            
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    st.session_state.documents = app.process_files(uploaded_files)
                    if context_files:
                        st.session_state.context_files = app.process_files(context_files)
                        st.session_state.context = ...

                    else:
                        st.session_state.context  = ""


                    # Prepare arguments for TldrEngine
                    args = {
                        'input_files': [f['path'] for f in st.session_state.documents],
                        'query': query,
                        'context_files': st.session_state.context,
                        'polish': polish,
                        'web_search': web_search
                    }
                    
                    # Initialize TldrEngine
                    st.session_state.tldr = TldrEngine(**args)


                    # TODO: generate component summaries at upload


        



        # Display file list
        if st.session_state.documents:
            st.subheader("Uploaded Files")
            for doc in st.session_state.documents:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(doc['name'])
                        st.caption(f"{doc['size']}")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['name']}"):
                            st.session_state.documents = [d for d in st.session_state.documents if d['name'] != doc['name']]
                            st.rerun()
        
        # Query input
        st.subheader("Query")
        query = st.text_area(
            "What would you like to know from these documents?",
            placeholder="Enter your query here...",
            height=100
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Generate Summary", disabled=not st.session_state.documents or app.processing):
                asyncio.run(app.generate_summary(query))
        with col2:
            if st.button("Polish", disabled=not st.session_state.summary or app.processing):
                asyncio.run(app.generate_summary(query, polish=True))
        with col3:
            if st.button("Research", disabled=app.processing):
                asyncio.run(app.generate_summary(query, web_search=True))
    
    with col2:
        st.subheader("Summary")
        
        # Summary tabs
        tab1, tab2 = st.tabs(["Current", "Polished"])
        
        # Summary content
        with tab1:
            summary = st.text_area(
                "Summary",
                value=st.session_state.summary,
                height=400,
                key="summary_output"
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
        disabled_attr = 'disabled' if not st.session_state.summary else ''
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
                    disabled=not st.session_state.summary,
                    help="Copy summary to clipboard"):
            st.session_state.clipboard = st.session_state.summary
            st.toast("Copied to clipboard!")
        
        if st.button("💾 Save as PDF",
                   key="pdf_btn_hidden",
                   disabled=not st.session_state.summary,
                   help="Save summary as PDF"):
            # TODO: Implement PDF saving
            st.toast("PDF saved!")
    
    # Status bar
    status_bar = st.container()
    with status_bar:
        st.markdown(
            f"""
            <div class='status-bar'>
                <span>Status: {app.status}</span>
                <span>Tokens: {app.token_count}</span>
                <span>v1.0.0</span>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    main()
