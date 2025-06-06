
import pickle

from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings


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
