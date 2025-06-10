import numpy as np
from typing import Dict, List, Any
from scipy.special import logsumexp

from .logger import ExpenseTracker


class DartboardRetriever(ExpenseTracker):
    def __init__(
        self,
        chunks: Any,
        default_num_results: int = 5,
        default_oversampling_factor: int = 3,
        default_div_weight: float = 1.0,
        default_rel_weight: float = 1.0,
        default_sigma: float = 0.1,
    ):
        super().__init__()
        """
        Initializes the DartboardRetriever.

        Args:
            chunks: The vector store (e.g., FAISS instance) containing document embeddings.
            default_num_results: Default number of documents to return.
            default_oversampling_factor: Default oversampling factor.
            default_div_weight: Default weight for diversity.
            default_rel_weight: Default weight for relevance.
            default_sigma: Default smoothing parameter for probability distribution.
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

        self.chunks = chunks
        self.default_num_results = default_num_results
        self.default_oversampling_factor = default_oversampling_factor
        self.default_div_weight = default_div_weight
        self.default_rel_weight = default_rel_weight
        self.default_sigma = default_sigma

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
        (Helper method, uses self.chunks)
        """
        docstore_id = self.chunks.index_to_docstore_id[idx]
        document = self.chunks.docstore.search(docstore_id)
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
        (Helper method)
        """
        sigma = max(sigma, 1e-5)
        query_probabilities = self._lognorm(query_distances, sigma)
        document_probabilities = self._lognorm(document_distances, sigma)

        most_relevant_idx = np.argmax(query_probabilities)
        # Corrected initialization based on your latest code
        selected_indices_for_masking = np.array([most_relevant_idx])
        selected_documents = [documents[most_relevant_idx]]
        selection_scores = [1.0]

        max_distances = document_probabilities[most_relevant_idx]

        # Ensure num_results is not greater than available documents
        num_results = min(num_results, len(documents))

        while len(selected_documents) < num_results:
            if len(selected_documents) >= len(
                documents
            ):  # Break if we've selected all available docs
                break

            # Create a mask for document_probabilities rows
            # This needs to be based on the indices of documents already selected
            # from the 'documents' list passed to this function.
            # We need to map indices from 'selected_documents' back to 'documents' if they differ.
            # For simplicity, assuming 'documents' here is the candidate_texts from get_context.
            # The 'selected_indices_for_masking' refers to indices within the current candidate set.

            # Update maximum distances considering new document
            # Ensure broadcasting is correct if document_probabilities is 1D for a single query_prob
            if (
                document_probabilities.ndim == 1
            ):  # If only one candidate doc initially (edge case)
                updated_distances = np.maximum(max_distances, document_probabilities)
            else:
                updated_distances = np.maximum(
                    max_distances[:, np.newaxis], document_probabilities
                )

            combined_scores = (
                updated_distances * div_weight
                + query_probabilities
                * rel_weight  # query_probabilities might need broadcasting if it's 1D
            )

            # Ensure combined_scores is 2D for logsumexp if not already
            if combined_scores.ndim == 1:
                combined_scores = combined_scores[:, np.newaxis]

            normalized_scores = logsumexp(combined_scores, axis=1)
            normalized_scores[selected_indices_for_masking] = (
                -np.inf
            )  # Mask already selected

            if np.all(
                np.isinf(normalized_scores)
            ):  # All remaining options are masked or invalid
                break

            best_idx = np.argmax(normalized_scores)
            best_score = np.max(normalized_scores)

            max_distances = updated_distances[best_idx]
            selected_documents.append(documents[best_idx])
            selection_scores.append(best_score)
            selected_indices_for_masking = np.append(
                selected_indices_for_masking, best_idx
            )

        return dict(zip(selected_documents, selection_scores))

    def get_embedded_context(
        self,
        query: str,
        num_results: int = None,
        oversampling_factor: int = None,
        div_weight: float = None,
        rel_weight: float = None,
        sigma: float = None,
    ) -> Dict[str, float]:
        """
        Retrieve most relevant and diverse context items for a query.
        Uses default parameters from __init__ if not provided.
        """
        num_results = (
            num_results if num_results is not None else self.default_num_results
        )
        oversampling_factor = (
            oversampling_factor
            if oversampling_factor is not None
            else self.default_oversampling_factor
        )
        div_weight = div_weight if div_weight is not None else self.default_div_weight
        rel_weight = rel_weight if rel_weight is not None else self.default_rel_weight
        sigma = sigma if sigma is not None else self.default_sigma

        # Ensure k for FAISS search is at least num_results
        k_search = max(num_results * oversampling_factor, num_results)
        if k_search == 0:  # Handle case where num_results is 0
            return {}

        query_embedding = self.chunks.embedding_function.embed_documents([query])

        # Check if the index is empty or k_search is too large
        if self.chunks.index.ntotal == 0:
            return {}
        k_search = min(k_search, self.chunks.index.ntotal)

        scores, candidate_indices = self.chunks.index.search(
            np.array(query_embedding), k=k_search
        )

        if not candidate_indices[0].size:  # No candidates found
            return {}

        candidate_vectors = np.array(
            self.chunks.index.reconstruct_batch(candidate_indices[0])
        )
        candidate_texts = [self._idx_to_text(idx) for idx in candidate_indices[0]]

        if not candidate_texts:  # No text for candidates
            return {}

        document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
        # query_distances needs to be 1D array of shape (n_candidates,)
        query_distances = (
            1 - np.dot(np.array(query_embedding), candidate_vectors.T)
        ).flatten()

        return self._greedy_dartsearch(
            query_distances,
            document_distances,
            candidate_texts,
            num_results,
            div_weight,
            rel_weight,
            sigma,
        )
