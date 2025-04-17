
import os
import re
import random
import importlib
from datetime import datetime

import requests
from requests.auth import HTTPBasicAuth

from docx import Document
from PyPDF2 import PdfReader, PdfMerger

from google.oauth2 import service_account
from googleapiclient.discovery import build

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from agents import function_tool


#--- MAIN -----------------------------------------------------------------------------#


@function_tool
def extract_code_snippets(message):
    """
    Extract code snippets from a large body of text using triple backticks as delimiters.
    Also saves the language tag at the start of each snippet.
    """
    # Regular expression to match code blocks enclosed in triple backticks, including the language tag
    code_snippets = defaultdict(str)
    code_pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
    snippets = code_pattern.findall(message)
    for lang, code in snippets:
        code_snippets[lang] += code.strip()

    return code_snippets


@function_tool
def write_script(content, lang, outDir="code"):
    """Writes code to a file."""
    os.makedirs(outDir, exist_ok=True)
    file_name = _select_object_name(content, lang) + extDict.get(lang, f".{lang}")
    with open(os.path.join(outDir, file_name), "w", encoding="utf-8") as f:
        f.write(f"#!/usr/bin/env {lang}\n\n{content}")


@function_tool
def import_python_module(file_path: str):
    """Import python modules from AI generated code"""

    if os.path.splitext(file_path)[1].lower() == '.py' and os.path.exists(file_path):
        module_name = file_path.replace('/','.')[0:-3]
        try:
            module = importlib.import_module(module_name)
            for import_name in dir(module):
                if not import_name.startswith('_'):  # Avoid importing private attributes
                    globals()[import_name] = getattr(module, import_name)
                    print(f"\tImported {import_name} to current environment.")
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")


@function_tool
def find_readable_files(directory):
    """
    Recursively scan the given directory and return a list of file paths
    that are likely to contain readable text, including .pdf and .docx files.
    """
    text_readable_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext in ['.pdf']:
                    # Try reading text from PDF
                    reader = PdfReader(filepath)
                    if any(page.extract_text() for page in reader.pages):
                        text_readable_files.append(filepath)

                elif ext in ['.docx']:
                    # Try reading text from DOCX
                    doc = Document(filepath)
                    if any(para.text.strip() for para in doc.paragraphs):
                        text_readable_files.append(filepath)

                else:
                    # Try opening as UTF-8 plain text
                    with open(filepath, 'r', encoding='utf-8') as f:
                        f.read(1024)
                    text_readable_files.append(filepath)

            except Exception:
                continue  # Skip unreadable or corrupted files

    return text_readable_files


@function_tool
def merge_pdf_library(directory: str = None):
    '''Concatenate PDFs in a given directory.'''
    files = os.listdir(directory) if directory else os.listdir()

    # Check if there are any PDF files in the directory
    pdf_files = [file for file in files if file.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    # Loop through all PDF files and append them to the merger
    merger = PdfMerger()
    for pdf_file in pdf_files:
        try:
            merger.append(pdf_file)
            print(f"Added {pdf_file} to the merger.")
        except Exception as e:
            print(f"Failed to add {pdf_file}: {e}")

    # Write out the concatenated PDF
    output_filename = f"merged.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pdf"
    try:
        merger.write(output_filename)
        print(f"All PDFs have been concatenated into {output_filename}")
    except Exception as e:
        print(f"Failed to write the concatenated PDF: {e}")
    finally:
        merger.close()

    return output_filename


@function_tool
def read_google_doc(doc_id, credentials_path):
    """
    Reads a Google Docs file and returns contents as a string

    Parameters:
        doc_id (str): The Google Docs file ID (from the URL).
        credentials_path (str): Path to the Google Cloud service account JSON credentials.

    Returns:
        str: The extracted text from the Google Doc.
    """
    # Authenticate using service account credentials
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/documents.readonly"]
    )
    
    # Build Google Docs API service
    service = build("docs", "v1", credentials=creds)
    
    # Retrieve the document
    document = service.documents().get(documentId=doc_id).execute()
    
    # Extract text from document structure
    text = []
    for element in document.get("body", {}).get("content", []):
        if "paragraph" in element:
            for run in element["paragraph"].get("elements", []):
                if "textRun" in run:
                    text.append(run["textRun"]["content"])
    
    return "\n".join(text)


@function_tool
def scrape_confluence_text(url: str, page_id: str, username: str = None, api_token: str = None) -> str:
    """
    Scrapes text from a Confluence page and returns a string.
    
    username: Confluence username (typically an email)
    api_token: API token for authentication
    """

    if username is not None and api_token is not None:
        response = requests.get(url, auth=(username, api_token), headers={"Accept": "application/json"})
    else:
        response = requests.get(url, headers={"Accept": "application/json"})
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page. Status code: {response.status_code}")
    
    data = response.json()
    html_content = data.get("body", {}).get("storage", {}).get("value", "")
    
    if not html_content:
        raise Exception("No content found on the page.")
    
    # Parse HTML content to extract text
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator='\n').strip()
    
    return text

@function_tool
def scrape_webpage_text(url, auth=None, headers=None, use_playwright=False, proxies=None, timeout=10):
    """
    Scrapes text content from a webpage using requests + BeautifulSoup, with optional Playwright for JavaScript-rendered pages.

    Parameters:
        url (str): The URL of the webpage.
        auth (tuple or dict, optional): Authentication credentials.
            - Tuple: (username, password) for HTTP Basic Auth.
            - Dict: Headers for token-based auth, e.g., {"Authorization": "Bearer <TOKEN>"}.
        headers (dict, optional): Additional HTTP headers.
        use_playwright (bool, optional): Whether to use Playwright for JavaScript-rendered pages.
        proxies (dict, optional): Proxy configuration, e.g., {"http": "http://proxy.com", "https": "https://proxy.com"}.
        timeout (int, optional): Request timeout in seconds (default: 10).

    Returns:
        str: Extracted text content or an error message.
    """
    # Use a random user-agent to avoid detection
    USER_AGENTS = [
        # Chrome (Windows)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.61",

        # Chrome (Mac)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",

        # Chrome (Linux)
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

        # Firefox (Windows)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0",
        
        # Firefox (Mac)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:118.0) Gecko/20100101 Firefox/118.0",

        # Safari (Mac)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        
        # Safari (iPhone)
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",

        # Safari (iPad)
        "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",

        # Android Chrome
        "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",

        # Android Firefox
        "Mozilla/5.0 (Android 13; Mobile; rv:119.0) Gecko/119.0 Firefox/119.0",
    ]
    final_headers = headers or {}
    final_headers["User-Agent"] = random.choice(USER_AGENTS)

    try:
        if use_playwright:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(extra_http_headers=final_headers)
                page.goto(url, timeout=timeout * 1000)  # Convert seconds to milliseconds
                html = page.content()  # Get full rendered HTML
                browser.close()
        else:
            # Handle authentication
            auth_param = None
            if isinstance(auth, tuple):
                auth_param = HTTPBasicAuth(*auth)  # HTTP Basic Auth
            elif isinstance(auth, dict):
                final_headers.update(auth)  # Token-based auth

            response = requests.get(url, headers=final_headers, auth=auth_param, proxies=proxies, timeout=timeout)
            response.raise_for_status()  # Raise HTTP errors

        # Parse HTML with BeautifulSoup for structured text extraction
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


#--- SUPPORT --------------------------------------------------------------------------#


def _select_object_name(code, language):
    """Extract defined object names from a code snippet"""

    # Get language-specific patterns
    patterns = patternDict.get(language, {})

    # Extract object names using the language-specific patterns
    classes = patterns.get("class", re.compile(r"")).findall(code)
    functions = patterns.get("function", re.compile(r"")).findallS(code)
    variables = patterns.get("variable", re.compile(r"")).findall(code)

    # Select objects to return based on hierarachy
    if len(classes) > 0:
        return max(classes, key=len)
    elif len(functions) > 0:
        return max(functions, key=len)
    else:
        return max(variables, key=len)


# File extension dictionary (do not match language directly)
extDict = {
   'bash': '.sh',
   'cuda': '.cu',
   'cython': '.pyx',
   'c++': '.cpp',
   'javascript':'.js',
   'julia':'.jl',
   'markdown': '.md',
   'matlab': '.mat',
   'nextflow': '.nf',
   'perl': '.pl',
   'python': '.py',
   'ruby': '.rb',
   'shell': '.sh',
   'text':'.txt',
   'plaintext': '.txt',
   }


# Code object patterns
patternDict = {
   "python": {
      "function": re.compile(r'def\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[:\(]'),
      "variable": re.compile(r'(\w+)\s*=\s*[^=\n]+'),
   },
   "javascript": {
      "function": re.compile(r'function\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[{]'),
      "variable": re.compile(r'(?:let|const|var)\s+(\w+)\s*='),
   },
   "java": {
      "function": re.compile(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[{]'),
      "variable": re.compile(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*='),
   },
   "r": {
      "function": re.compile(r'(\w+)\s*<-\s*function\s*\('),
      "variable": re.compile(r'(\w+)\s*<-\s*[^=\n]+'),
   },
   "groovy": {
      "function": re.compile(r'def\s+(\w+)\s*\('),
      "class": re.compile(r'class\s+(\w+)\s*[{]'),
      "variable": re.compile(r'def\s+(\w+)\s*='),
   },
}

