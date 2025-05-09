from scipy.special import logsumexp
import numpy as np
from typing import Tuple, List, Any


def greedy_dartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int = 10,
    div_weight: float = 1.0, # Weight for diversity in document selection
    rel_weight: float = 1.0, # Weight for relevance to query
    sigma: float = 0.1 # Smoothing parameter for probability distribution
) -> Tuple[List[str], List[float]]:
    """
    Perform greedy dartboard search to select top k documents balancing relevance and diversity.
    
    Args:
        query_distances: Distance between query and each document
        document_distances: Pairwise distances between documents
        documents: List of document texts
        num_results: Number of documents to return
    
    Returns:
        Tuple containing:
        - List of selected document texts
        - List of selection scores for each document
    """
    # Avoid division by zero in probability calculations
    sigma = max(sigma, 1e-5)
    
    # Convert distances to probability distributions
    query_probabilities = lognorm(query_distances, sigma)
    document_probabilities = lognorm(document_distances, sigma)
    
    # Initialize with most relevant document
    
    most_relevant_idx = np.argmax(query_probabilities)
    selected_indices = np.array([most_relevant_idx])
    selection_scores = [1.0] # dummy score for the first document
    # Get initial distances from the first selected document
    max_distances = document_probabilities[most_relevant_idx]
    
    # Select remaining documents
    while len(selected_indices) < num_results:
        # Update maximum distances considering new document
        updated_distances = np.maximum(max_distances, document_probabilities)
        
        # Calculate combined diversity and relevance scores
        combined_scores = (
            updated_distances * div_weight +
            query_probabilities * rel_weight
        )
        
        # Normalize scores and mask already selected documents
        normalized_scores = logsumexp(combined_scores, axis=1)
        normalized_scores[selected_indices] = -np.inf
        
        # Select best remaining document
        best_idx = np.argmax(normalized_scores)
        best_score = np.max(normalized_scores)
        
        # Update tracking variables
        max_distances = updated_distances[best_idx]
        selected_indices = np.append(selected_indices, best_idx)
        selection_scores.append(best_score)
    
    # Return selected documents and their scores
    selected_documents = [documents[i] for i in selected_indices]
    return selected_documents, selection_scores

def lognorm(dist:np.ndarray, sigma:float):
    """
    Calculate the log-normal probability for a given distance and sigma.
    """
    if sigma < 1e-9: 
        return -np.inf * dist
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)


def get_context_with_dartboard(
    query: str,
    chunks: Any,
    k: int = 3,
    num_results: int = 5,
    oversampling_factor: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Retrieve most relevant and diverse context items for a query using the dartboard algorithm.
    
    Args:
        query: The search query string
        num_results: Number of context items to return (default: 5)
        oversampling_factor: Factor to oversample initial results for better diversity (default: 3)
    
    Returns:
        Tuple containing:
        - List of selected context texts
        - List of selection scores
        
    Note:
        The function uses cosine similarity converted to distance. Initial retrieval 
        fetches oversampling_factor * num_results items to ensure sufficient diversity 
        in the final selection.
    """
    # Embed query and retrieve initial candidates
    query_embedding = chunks.embedding_function.embed_documents([query])
    _, candidate_indices = chunks.index.search(
        np.array(query_embedding),
        k=num_results * oversampling_factor
    )
    
    # Get document vectors and texts for candidates
    candidate_vectors = np.array(
        chunks.index.reconstruct_batch(candidate_indices[0])
    )
    candidate_texts = [idx_to_text(idx) for idx in candidate_indices[0]]
    
    # Calculate distance matrices
    # Using 1 - cosine_similarity as distance metric
    document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
    query_distances = 1 - np.dot(query_embedding, candidate_vectors.T)
    
    # Apply dartboard selection algorithm
    selected_texts, selection_scores = greedy_dartsearch(
        query_distances,
        document_distances,
        candidate_texts,
        num_results
    )
    
    return selected_texts[0:k], selection_scores[0:k]

def idx_to_text(idx, chunks):
    """
    Convert a Vector store index to the corresponding text.
    """
    docstore_id = chunks.index_to_docstore_id[idx]
    document = chunks.docstore.search(docstore_id)
    return document.page_content

