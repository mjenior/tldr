# tldr

A powerful text summarization tool that uses OpenAI's models to generate concise summaries and evaluations of scientific documents.

Version = 1.0.0

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

## Installation

This project uses `pixi` for environment and dependency management.

```bash
# Install from source
git clone https://github.com/mattjenior/tldr.git
cd tldr

# Install dependencies and activate the environment
pixi install
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

## Usage

### Command Line

The tool is primarily used via the command line.

```bash
# Basic usage - summarize all text files in the current directory
tldr

# Provide a specific query to guide the summarization
tldr "What are the latest advancements in CRISPR gene editing?"

# Specify input files for summarization
tldr -i ./my_papers/*

# Add context documents to inform the summary
tldr -c ./context_files/*

# Enable recursive file search in input directories
tldr -r

# Disable the web research agent
tldr -w False

# Use a modified output tone (e.g., 'technical' or 'casual')
tldr -t modified
```

### Available Arguments

*   `query` (Positional): Your question or topic of interest. (Default: None)
*   `-q`, `--refine_query <True|False>`: Automatically refine the user query. (Default: `True`)
*   `-i`, `--input_files <path>`: List of input files or directories. (Default: None)
*   `-c`, `--context_files <path>`: List of files for additional context. (Default: None)
*   `-r`, `--recursive_search <True|False>`: Recursively search input directories. (Default: `False`)
*   `-w`, `--web_search <True|False>`: Use the research agent to fill knowledge gaps. (Default: `True`)
*   `-t`, `--tone <tone>`: Define the tone for the final summary. (Default: `default`)
*   `-s`, `--context_size <scale>`: Set the context window size for the model. (Choices: `low`, `medium`, `high`; Default: `medium`)
*   `-v`, `--verbose <True|False>`: Enable verbose logging to stdout. (Default: `True`)

### Summary Quality Evaluator (Python API)

You can programmatically evaluate the quality of a generated summary using the `SummaryJudge` class.

```python
from tldr.judge import SummaryJudge

# Define paths to the original content and the generated summary
content_path = "path/to/original_document.txt"
summary_path = "path/to/generated_summary.txt"

# Instantiate the evaluator
judge = SummaryJudge()

# Generate an objective score from file paths
judge.evaluate(content_path=content_path, summary_path=summary_path)

# You can also pass content and summary strings directly
# judge.evaluate(content=content_str, summary=summary_str)

# Print the results
print('Evaluation Iterations:', '\n\n'.join(judge.evaluation_iters))
print('\nFinal Evaluation:', judge.evaluations)
```

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

