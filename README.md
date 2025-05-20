# tldr

A powerful text summarization tool that uses OpenAI's models to generate concise summaries and evaluations of scientific documents.

Version = 0.2.3

## Overview

`tldr` is a Python package designed to help researchers, scientists, and knowledge workers quickly extract key information from scientific publications, technical documents, and software REAME files. It uses OpenAI's language models to:

1. Scan and process multiple document types (PDF, DOCX, HTML, TXT)
2. Generate detailed summaries of each document
3. Synthesize information across multiple documents
4. Perform additional research to fill knowledge gaps
5. Create polished, comprehensive responses to user queries

## Features

- **Multi-format document support**: PDF, DOCX, HTML, MD, and TXT files
- **Intelligent query refinement**: Automatically improves user queries
- **Document summarization**: Creates detailed summaries of scientific publications and technical documents
- **Cross-document synthesis**: Integrates information across multiple sources
- **Knowledge gap research**: Automatically identifies and researches missing information
- **Customizable output**: Supports different output formats and tones
- **Glyph-based synthesis**: Optional advanced synthesis mode using symbolic representation
- **Objective summary evaluation**: More objectively evaluate the quality of individual output summaries

## Installation

```bash
# Install from source
git clone https://github.com/mattjenior/tldr.git
cd tldr
pip install -e .
```

## Requirements

- Python 3.11 or higher
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

### Packages:

- openai >= 1.78.1
- pillow >= 10.2.0
- pyyaml >= 6.0.1
- jsonschema >= 4.21.1
- python-docx
- pypdf2
- beautifulsoup4
- black

## Usage

### Available Arguments

*   `query` (Positional): Optional user query. (Default: None)
*   `-i, --input_directory <path>`: Directory for input text files. (Default: `.`)
*   `-o, --output_directory <path>`: Directory for output files. (Default: `.`)
*   `-r, --refine_query <True|False>`: Automatically refine user query. (Default: `False`)
*   `-v, --verbose <True|False>`: Verbose stdout reporting. (Default: `True`)
*   `-s, --research <True|False>`: Use research agent for knowledge gaps. (Default: `True`)
*   `-n, --tone <tone>`: Final summary tone. (Choices: `default`, `modified`; Default: `default`)
*   `-g, --glyphs <True|False>`: Utilize associative glyphs in summary. (Default: `False`)
*   `-t, --token_scale <scale>`: Scale for max output tokens. (Choices: `low`, `medium`, `high`; Default: `medium`)
*   `-c, --context_size <scale>`: Context window size for research agent web search. (Choices: `low`, `medium`, `high`; Default: `medium`)

### Command Line

```bash
# Basic usage - summarize all text files in current directory
tldr

# Add a specific query
tldr "What are the latest advancements in CRISPR gene editing?"

# Enable automatic query refinement
tldr "CRISPR advances" -r True

# Specify input directory and output directory
tldr -i ./my_papers -o ./summaries

# Enable query prompt refinement
tldr -r True

# Disable research agent
tldr -s False

# Use modified output tone
tldr -n modified

# Enable glyph-based synthesis
tldr -g True

# Increase the scale of max tokens allowed per response
tldr -t long

# Increase conext window size for gap-filling web search
tldr -c high
```

### Python API

```python
import asyncio
from tldr import TldrClass

# Initialize with a user query
tldr = TldrClass(search_directory="./my_papers", output_directory="./summaries")

# Refine the user query
tldr.refine_user_query("What are the latest advancements in CRISPR gene editing?")

# Generate document summaries
tldr.all_summaries = asyncio.run(tldr.summarize_resources())

# Integrate summaries across documents
tldr.integrate_summaries()

# Research additional information
tldr.apply_research()

# Polish the final response
tldr.polish_response()
```

### Summary Quality Evaluator

```python
from tldr.summary_judge import SummaryJudge

# Define paths and inputs
content_path = "path/to/original_technical_text.txt"
generated_summary_path = "path/to/summary_text.txt"

# Instantiate the evaluator
judge = SummaryEvaluator()

# Generate objective scoring from file paths
judge.evaluate(content_path=content_path, summary_path=generated_summary_path)
# Also can handle content and summary strings variables directly
judge.evaluate(content=content_str, summary=generated_summary_str)

# Print results
print('Iterations:', '\n\n'.join(judge.evaluation_iters))
print('\nFinal Evaluation:\n', judge.evaluations)

```

## Output Files

The tool generates several output files during processing:

- `tldr.summaries.[file_count].[timestamp].txt`: Individual document summaries
- `tldr.synthesis.[timestamp].txt`: Integrated summary across documents
- `tldr.research.[timestamp].txt`: Additional research information
- `tldr.final.[timestamp].txt`: Final polished response

## Configuration

Configuration is handled through command-line arguments or directly via the API. The system uses a set of carefully crafted prompts defined in `instructions.yaml`.

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Author

- Matthew Jenior (mattjenior@gmail.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
