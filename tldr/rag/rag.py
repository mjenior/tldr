"""
Evaluates the performance of a Retrieval-Augmented Generation (RAG) system
"""

from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


import pickle

def save_embedding(embedding, pkl_name):
    """Save a vector library locally."""
    try:
        with open(f"{pkl_name}.pkl", "wb") as f: 
            pickle.dump(embedding, f)
    except Exception as e:
        print(f"An error occurred saving embedding: {e}")

        
def evaluate_rag(user_request, retriever, num_questions: int = 10) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.
    
    Args:
        user_request: User request string
        retriever: The retriever component to evaluate
        num_questions: Number of test questions to generate
    
    Returns:
        Dict containing evaluation metrics
    """

    # Initialize model
    model = ChatOpenAI(temperature=0.5, model_name="gpt-4o")

    # Generate test questions
    question_prompt = PromptTemplate.from_template("""
    Generate {num_questions} diverse test questions that address the following user request:
    {user_request}
    """)
    question_chain = RunnableSequence(question_prompt, model, StrOutputParser())
    questions = question_chain.invoke({"num_questions": num_questions, "user_request": user_request}).split("\n")
    
    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Correctness: Is factually correct based on the expected output?
    2. Relevance: How relevant is the retrieved information to the question?
    3. Completeness: Does the context contain all necessary information?
    4. Faithfulness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """)
    eval_chain = RunnableSequence(eval_prompt, model, StrOutputParser())

    # Evaluate each question
    results = []
    for question in questions:
        # Get retrieval results
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)
    
    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }


def calculate_average_scores(results: List[Dict[str, float]]) -> Dict[int, float]:
    """
    Calculates the mean of values for each dictionary in a list and returns
    a new dictionary mapping the original index to its calculated mean.
    """
    mean_results: Dict[int, float] = {}
    index: int = 0

    # Enumerate provides both the index and the item
    for current_dict in results:
        index += 1
        values = [x for list(current_dict.values()) if isinstance(x, float)]

        # Check if the dictionary has any values to average
        if not values:
            mean_value = 0.0
        else:
            mean_results[f"result_{index}"] = sum(values) / len(values)

    return mean_results
