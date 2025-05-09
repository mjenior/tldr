
import os
import yaml
from datetime import datetime

import argparse
from argparse import Namespace


def parse_tldr_arguments() -> Namespace:
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(description="TLDR: Summarize text files based on user arguments.")

    # Define arguments 
    parser.add_argument(
        'input_directory',
        nargs='?', # Makes the argument optional
        default='.',
        help='Directory to scan for text files (Default is working directory)')
    parser.add_argument(
        '-o', '--output_directory',
        default='.',
        help='Directory for output files (Default is working directory)')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', # This makes the argument a boolean flag
        help='Enable verbose output')
    parser.add_argument(
        '-q', '--query',
        type=str,
        default=None,
        help='Optional user query')
    parser.add_argument(
        '-r', '--research',
        action='store_true', 
        help='Addition research agent.')
    parser.add_argument(
        '-t', '--tone',
        choices=['formal', 'casual'],
        default='formal',
        help='Response tone.')
    parser.add_argument(
        '-s', '--summary_type',
        choices=['document', 'lesson', 'podcast'],
        default='document',
        help='Response type.')

    return parser.parse_args()


def system_instructions(self, file_path: str = 'instructions.yaml') -> dict:
    """
    Reads a YAML file and returns its content as a Python dictionary.
    """
    tldr_path = os.path.dirname(os.path.abspath(__file__))
    instructions_path = os.path.join(tldr_path, file_path)
    try:
        with open(instructions_path, 'r') as file:
            data = yaml.safe_load(file)
            if isinstance(data, dict):
                self.instructions = data
            else:
                print(f"Error: Content in '{instructions_path}' is not a dictionary (YAML root is type: {type(data)}).")
    except Exception as e:
        print(f"An unexpected error occurred while reading '{instructions_path}': {e}")



def save_response_text(data_str: str, label: str = "response", output_dir: str = ".") -> str:
    """
    Saves a large string variable to a text file with a dynamic filename
    based on a timestamp and a user-provided label.
    """
    errors = 'strict'
    chunk_size = 1024*1024

    # Generate a timestamp string (e.g., 20231027_103000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize the label to create a valid filename part
    # Replace spaces with underscores and remove characters not suitable for filenames
    sanitized_label = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in label).strip('_')

    # Construct the dynamic filename
    filename = f"{sanitized_label}.{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filename, 'w', encoding='utf-8', errors=errors) as f:
            # Write in chunks to avoid excessive memory usage for very large strings
            for i in range(0, len(data_str), chunk_size):
                # Note: Corrected variable name from data_string to data_str
                f.write(data_str[i:i + chunk_size])

        print(f"Saved data to {filename}")

    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
        raise  # Re-raise the exception after printing
    except Exception as e:
        print(f"An unexpected error occurred while saving {filename}: {e}")
        raise # Re-raise the exception