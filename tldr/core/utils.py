
import yaml 
from typing import Any, Optional


def read_yaml_config(file_path: str) -> Optional[Any]:
    """
    Reads and parses data from a specified YAML file using safe_load.

    Args:
        file_path: The path (string) to the YAML file.

    Returns:
        The Python object representation of the YAML data (e.g., dict, list),
        or None if an error occurs during file access or YAML parsing.
        Errors encountered during the process are printed to standard error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise Exception(f"Error: Issue occurred while processing '{file_path}'. {e}")
        return None
