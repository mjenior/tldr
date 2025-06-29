import numpy as np
from typing import Dict, List
import asyncio
from scipy.special import logsumexp
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from .logger import LogHandler


class ResearchAgent(LogHandler):
    """
    A class for performing research tasks, including document retrieval and ranking.

    This agent is designed to find the most relevant and diverse set of documents
    from a given corpus based on a user query. It uses techniques like Maximal
    Marginal Relevance (MMR) to balance relevance and diversity in the search
    results.

    Attributes:
        num_results (int): The default number of documents to return.
        oversampling_factor (int): A factor to determine how many initial
            candidates to retrieve before ranking.
        div_weight (float): The weight given to diversity in the MMR calculation.
        rel_weight (float): The weight given to relevance in the MMR calculation.
        sigma (float): A smoothing parameter for the probability distribution
            used in the ranking algorithm.
    """

    def __init__(
        self,
        num_results: int = 5,
        oversampling_factor: int = 3,
        div_weight: float = 1.0,
        rel_weight: float = 1.0,
        sigma: float = 0.1,
        platform: str = "openai",
    ):
        super().__init__()
        """
        Initializes the ResearchAgent.

        Args:
            num_results: Default number of documents to return.
            oversampling_factor: Default oversampling factor.
            div_weight: Default weight for diversity.
            rel_weight: Default weight for relevance.
            sigma: Default smoothing parameter for probability distribution.
            platform: Default platform for embeddings (OpenAI vs Gemini)
        """
        self.num_results = num_results
        self.oversampling_factor = oversampling_factor
        self.div_weight = div_weight
        self.rel_weight = rel_weight
        self.sigma = sigma
        self.platform = platform

    async def encode_text_to_vector_store(
        self,
        extra_text: str = "",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        platform: str = "openai",
    ) -> FAISS:
        """
        Encodes a string into a FAISS vector store using OpenAI embeddings.
        """
        self.logger.info("Generating embeddings for reference documents...")
        chunk_size = 1000 if chunk_size <= 0 else chunk_size
        chunk_overlap = 200 if chunk_overlap < 0 else chunk_overlap

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            # Handle the content properly
            content_text = self.join_text_objects(self.content)

            # Add extra text if provided
            content_text = f"{content_text}\n\n{extra_text}".strip()
                
            # Ensure we have content to process
            if not content_text:
                raise ValueError("No content available to encode into vector store")
                
            # Create documents - wrap in list as create_documents expects a list of strings
            docs = text_splitter.create_documents([content_text])

            # Assign metadata to each chunk (Langchain Document objects)
            for doc in docs:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["relevance_score"] = 1.0 # baseline relevance

            embeddings = (
                OpenAIEmbeddings()
                if platform == "openai"
                else GoogleGenerativeAIEmbeddings()
            )
            vector_store = FAISS.from_documents(docs, embeddings)
            self.check_vector_store(vector_store)
            
            return vector_store

        except Exception as e:
            self.logger.error(f"Error in encode_text_to_vector_store: {str(e)}")
            raise RuntimeError(
                f"An error occurred during the encoding process: {str(e)}"
            )
    
    @staticmethod
    def join_text_objects(collection, separator="\n\n", data_type="values"):
        """Joins a collection of text objects into a single string.
        
        Args:
            collection: A string, dictionary, or iterable of strings/dictionaries to join
            separator: String to place between each item (default: "\n\n")
            data_type: "values" or "keys" (default: "values")
            
        Returns:
            str: A single string with all items joined by the separator
            
        Raises:
            TypeError: If any item cannot be converted to string
        """
        def to_strings(items):
            result = []
            for item in items:
                if isinstance(item, dict):
                    if data_type == "values":
                        result.extend(to_strings(item.values()))
                    elif data_type == "keys":
                        result.extend(to_strings(item.keys()))
                elif item is not None:
                    result.append(str(item))
            return [x.strip() for x in result]

        if collection is None:
            return ""
            
        if isinstance(collection, str):
            return collection.strip()
            
        if isinstance(collection, dict):
            collection = to_strings(collection)
            
        if not isinstance(collection, (list, tuple)):
            collection = [str(collection).strip()]
            
        return separator.join(collection)

    def check_vector_store(self, chunks: FAISS):
        """
        Add new vector store (e.g., FAISS instance) containing document embeddings.
        """
        if (
            not hasattr(chunks, "embedding_function")
            or not hasattr(chunks, "index")
            or not hasattr(chunks, "index_to_docstore_id")
            or not hasattr(chunks, "docstore")
        ):
            raise ValueError(
                "Provided 'chunks' object does not appear to be a compatible vector store (e.g., FAISS)."
            )
        else:
            self.vector_store = chunks

    def _lognorm(self, dist: np.ndarray, sigma: float) -> np.ndarray:
        """
        Calculate the log-normal probability for a given distance and sigma.
        (Helper method)
        """
        if sigma < 1e-9:
            return -np.inf * np.ones_like(
                dist
            )  # Ensure it returns an array of the same shape
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)

    def _idx_to_text(self, idx: int) -> str:
        """
        Convert a vector store index to the corresponding text.
        """
        docstore_id = self.vector_store.index_to_docstore_id[idx]
        document = self.vector_store.docstore.search(docstore_id)
        return document.page_content

    def _greedy_dartsearch(
        self,
        query_distances: np.ndarray,
        document_distances: np.ndarray,
        documents: List[str],
        num_results: int,
        div_weight: float,
        rel_weight: float,
        sigma: float,
    ) -> Dict[str, float]:
        """
        Perform greedy dartboard search.
        """
        num_results = min(num_results, len(documents))
        sigma = max(sigma, 1e-5)
        query_probabilities = self._lognorm(query_distances, sigma)
        document_probabilities = self._lognorm(document_distances, sigma)

        # Select the most relevant document
        most_relevant_idx = np.argmax(query_probabilities)
        selected_indices_for_masking = np.array([most_relevant_idx])
        selected_documents = [documents[most_relevant_idx]]
        selection_scores = [1.0]
        max_distances = document_probabilities[most_relevant_idx]

        for _ in range(num_results):
            # Exit loop if all docs read
            if len(selected_documents) >= len(documents):
                break

            # Create a mask for document_probabilities rows
            if document_probabilities.ndim == 1:
                updated_distances = np.maximum(max_distances, document_probabilities)
            else:
                updated_distances = np.maximum(
                    max_distances[:, np.newaxis], document_probabilities
                )
            combined_scores = (
                updated_distances * div_weight + query_probabilities * rel_weight
            )

            # Ensure combined_scores is 2D for logsumexp if not already
            if combined_scores.ndim == 1:
                combined_scores = combined_scores[:, np.newaxis]
            normalized_scores = logsumexp(combined_scores, axis=1)
            normalized_scores[selected_indices_for_masking] = -np.inf
            if np.all(np.isinf(normalized_scores)):
                break

            # Select the best document
            best_idx = np.argmax(normalized_scores)
            best_score = np.max(normalized_scores)
            max_distances = updated_distances[best_idx]
            selected_documents.append(documents[best_idx])
            selection_scores.append(best_score)
            selected_indices_for_masking = np.append(
                selected_indices_for_masking, best_idx
            )

        # Return the selected documents and their scores
        return dict(zip(selected_documents, selection_scores))

    async def search_embedded_context(self, query: str, num_results: int = 5) -> str:
        """
        Retrieve most relevant and diverse context items for a query.
        Uses default parameters from __init__ if not provided.
        """
        self.logger.info("Querying local embedded text vectors...")

        # Check if the index is empty or k_search is too large
        if self.vector_store.index.ntotal == 0:
            return {}
        # Ensure k for FAISS search is at least num_results
        k_search = max(num_results * self.oversampling_factor, num_results)
        if k_search == 0:
            return {}

        # Query local embedded text vectors
        query_embedding = self.vector_store.embedding_function.embed_documents([query])
        scores, candidate_indices = self.vector_store.index.search(
            np.array(query_embedding), k=min(k_search, self.vector_store.index.ntotal)
        )

        # Extract candidate text
        if not candidate_indices[0].size:
            return {}
        candidate_vectors = np.array(
            self.vector_store.index.reconstruct_batch(candidate_indices[0])
        )
        candidate_texts = [self._idx_to_text(idx) for idx in candidate_indices[0]]
        if not candidate_texts:
            return {}
        document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
        query_distances = (
            1 - np.dot(np.array(query_embedding), candidate_vectors.T)
        ).flatten()

        # Perform greedy dartboard search
        result = self._greedy_dartsearch(
            query_distances,
            document_distances,
            candidate_texts,
            num_results,
            self.div_weight,
            self.rel_weight,
            self.sigma,
        )
        self.logger.info(
            "Average relevance score: {}".format(
                round(np.mean(list(result.values())), 3)
            )
        )
        return self.join_text_objects(result, data_type="keys")
