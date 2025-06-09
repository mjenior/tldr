
import pickle
import argparse

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings


def parse_user_arguments():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="TLDR: Summarize text files based on user arguments."
    )

    # Define arguments
    parser.add_argument("query", nargs="?", default="", help="Optional user query")
    parser.add_argument(
        "-i",
        "--input_files",
        default=None,
        nargs="+",
        help="Optional: Input files to summarize (Default is scan for text files in working directory)",
    )
    parser.add_argument(
        "-f",
        "--refine_query",
        type=bool,
        default=True,
        help="Automatically refine and improve the user query",
    )
    parser.add_argument(
        "-c",
        "--context_files",
        default=None,
        nargs="+",
        help="Optional: Additional context documents to include in the system instructions",
    )
    parser.add_argument(
        "-r",
        "--recursive_search",
        type=bool,
        default=False,
        help="Recursively search input directories",
    )
    parser.add_argument(
        "-w",
        "--web_search",
        type=bool,
        default=True,
        help="Additional research agent to find and fill knowledge gaps",
    )
    parser.add_argument(
        "-t",
        "--tone",
        choices=["default", "modified"],
        default="default",
        help="Final executive summary response tone",
    )
    parser.add_argument(
        "-s",
        "--context_size",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Modifier for scale of maximum output tokens and context window size for research agent web search",
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=True, help="Verbose stdout reporting"
    )
    parser.add_argument(
        "--testing",
        type=bool,
        default=False,
        help="Uses only gpt-4o-mini for cheaper+faster testing runs.",
    )

    return vars(parser.parse_args())

def encode_text(content: str or list, file_label: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    Encodes a string into a FAISS vector store using OpenAI embeddings.
    """
    try:
        vectorstore = load_embedding(file_label)
    except Exception as e:
        content = content if isinstance(content, list) else content.split("\n")
        chunk_size = 1000 if chunk_size <= 0 else int(chunk_size)
        chunk_overlap = 200 if chunk_overlap < 0 else int(chunk_overlap) # overlap can be 0

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            # create_documents expects a list of strings
            docs = text_splitter.create_documents(content) 

            # Assign metadata to each chunk (Langchain Document objects)
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['relevance_score'] = 1.0
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save embedding
            save_embedding(vectorstore, file_label)

        except Exception as e:
            raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def save_embedding(embedding: FAISS, file_label: str):
    """Save a FAISS vector library locally."""
    try:
        with open(f"{file_label}.vector_lib.pkl", "wb") as f:
            pickle.dump(embedding, f)
    except Exception as e:
        print(f"An error occurred saving embedding: {e}")


def load_embedding(file_label: str) -> FAISS:
    """Load a FAISS vector library from a local pickle file."""
    try:
        with open(f"{file_label}.vector_lib.pkl", "rb") as f:
            embedding = pickle.load(f)
        return embedding
    except FileNotFoundError:
        print(f"Error: The file {file_label}.vector_lib.pkl was not found.")
        return None
    except Exception as e:
        print(f"An error occurred loading embedding: {e}")
        return None
