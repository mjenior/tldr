# tldr

A powerful text summarization tool that uses OpenAI's models to generate concise summaries and evaluations of scientific documents.

Version: 1.1.3

## Overview

`tldr` is a Python package designed to help researchers, scientists, and knowledge workers quickly extract key information from scientific publications, technical documents, and software README files. It streamlines the process of understanding complex information by leveraging the power of large language models.

At its core, `tldr` processes various document formats, generates detailed summaries, synthesizes information across multiple sources, and even performs web research to fill knowledge gaps, delivering a polished and comprehensive response to user queries.

## How It Works

The tool operates through a modular, object-oriented design:

-   **`TldrEngine`**: The central orchestrator. It manages the entire summarization pipeline, from parsing user arguments to generating the final polished response.
-   **`FileHandler`**: A robust module for all file I/O operations. It handles reading different document types (PDF, DOCX, HTML, TXT, Markdown), discovering readable files, and saving all output files, including intermediate summaries and the final PDF report.
-   **`CompletionHandler`**: Manages all interactions with the OpenAI API, including sending prompts and processing the responses for summarization, query refinement, and research.
-   **`SummaryJudge`**: An evaluation tool to objectively score the quality of generated summaries based on criteria like correctness, coherence, and conciseness.

This architecture allows for a clear separation of concerns and makes the system easily extensible.

## Features

-   **Multi-format document support**: Handles PDF, DOCX, HTML, Markdown, and TXT files.
-   **Intelligent Query Refinement**: Automatically improves user queries for more accurate results.
-   **Local RAG with Vector Store**: Creates local embeddings of your documents to provide highly relevant, in-context answers to your queries.
-   **Cross-document Synthesis**: Integrates information across multiple sources into a single, coherent summary.
-   **Automated Research Agent**: Identifies knowledge gaps in the source material and uses web searches to find the missing information.
-   **Customizable Output**: Supports different output tones and formats, including a final polished PDF report.
-   **Objective Summary Evaluation**: Provides tools to score the quality of the generated summaries.
-   **Prompt Injection Testing**: Tests the system for prompt injection vulnerabilities.

## Installation

This project uses [pixi](https://pixi.sh/latest/) for environment and dependency management through `make`.

```bash
# Install from source
git clone https://github.com/mattjenior/tldr.git
cd tldr
make install

# In most cases, installation can also be done with pip
pip install -e .
```

## Requirements

-   Python 3.11 or higher
-   An OpenAI API key, set as an environment variable `OPENAI_API_KEY`.

Key Python packages are listed in the `pixi.toml` file and include:

-   `openai`
-   `pypdf2`
-   `python-docx`
-   `beautifulsoup4`
-   `pyyaml`
-   `reportlab`
-   `chromadb`
-   `lancedb`
-   `numpy`
-   `deepeval`
-   `streamlit`
-   `faiss`
-   `tenacity`
-   `langchain`


## Usage

### Web Interface (Streamlit)

TLDR is most easily used via a web interface.

```bash
# Run the web interface
streamlit run interface.py

# Or simply use the tldr_ui alias added during installation
tldr_ui
```

### Command Line

The tool can also be used via the command line very simply.

```bash
# Basic usage - summarize all text files in the current directory
tldr

# Provide a specific query to guide the summarization
tldr "What are the latest advancements in CRISPR gene editing?"

# Specify input files for summarization
tldr --input_files ./my_papers/*

# Add context documents to inform the summary
tldr --context_files ./context_files/*

# Enable recursive file search in input directories
tldr --recursive_search True

# Disable the web research agent
tldr --web_search False

# Use a modified output tone (e.g., 'stylized' or 'formal')
tldr --tone formal
```

### Available Arguments

*   `query` (Positional): Your question or topic of interest. (Default: None)
*   `--refine_query <True|False>`: Automatically refine the user query. (Default: `True`)
*   `--input_files <path>`: List of input files or directories. (Default: None)
*   `--context_files <path>`: List of files for additional context. (Default: None)
*   `--recursive_search <True|False>`: Recursively search input directories. (Default: `False`)
*   `--web_search <True|False>`: Use the research agent to fill knowledge gaps. (Default: `True`)
*   `--tone <tone>`: Define the tone for the final summary. (Default: `default`)
*   `--context_size <scale>`: Set the context window size for the model. (Choices: `low`, `medium`, `high`; Default: `medium`)
*   `--verbose <True|False>`: Enable verbose logging to stdout. (Default: `True`)
*   `--api_key <key>`: API key for OpenAI. (Default: None)
*   `--injection_screen <True|False>`: Enable prompt injection testing. (Default: `True`)

### Summary Evaluation (Python API)

You can evaluate the quality of generated summaries using the Python API for easier iterative testing.

#### Python API

```python
from tldr.eval import SummaryEvaluator

# Initialize the evaluator with custom settings
evaluator = SummaryEvaluator(
    iterations=5,  # Number of evaluations to run (default: 5)
    model="gpt-4o-mini",  # Model to use for evaluation
    temperature=0.9,  # Controls randomness in evaluation
    verbose=True  # Enable detailed logging
)

# Evaluate a summary against its source content
evaluation = await evaluator.evaluate(
    content="path/to/original_document.txt",  # Can be file path or string
    summary="path/to/generated_summary.txt"    # Can be file path or string
)

# Access the evaluation results
print(f"Evaluation Score: {evaluation['score']}")
print(f"Reasoning: {evaluation['reason']}")
```

CLI Options:
- `--content`: Path to the original content file or string
- `--summary`: Path to the generated summary file or string
- `--model`: Model to use for evaluation (default: "gpt-4o-mini")
- `--iterations`: Number of evaluation iterations (default: 5)
- `--verbose`: Enable verbose output (default: True)

#### Evaluation Metrics

The evaluation uses two complementary approaches:

1. **Faithfulness Metric**: Measures factual consistency between the summary and source
   - Checks for hallucinated or inconsistent information
   - Returns a score between 0 and 1

2. **GEval**: Comprehensive quality assessment using LLM-based evaluation
   - Evaluates coherence, conciseness, completeness, and style
   - Provides detailed reasoning for the assigned score

All evaluations are saved in a timestamped directory for reference.

## Output Files

The tool generates several output files in a timestamped directory (`tldr.[timestamp].files/`):

-   `summary.[index].tldr.[timestamp].txt`: Individual summaries for each input document.
-   `synthesis.tldr.[timestamp].txt`: The integrated summary combining all source documents.
-   `research.tldr.[timestamp].txt`: Additional information gathered by the research agent.
-   `[content_based_name].tldr.pdf`: The final, polished response formatted as a PDF.
-   `tldr.[timestamp].log`: A log file detailing the entire process.

## Configuration

System behavior is configured through command-line arguments. The prompts used for interacting with the language model are defined in `tldr/instructions.yaml`.

## License

This project is licensed under the MIT License - see the `LICENSE.txt` file for details.

## Author

-   Matthew Jenior (mattjenior@gmail.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
