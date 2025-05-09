


def answer_query(self, prompt, context, model='gpt-4o'):
    """Processes text-based responses from OpenAIs chat models."""
    if isinstance(context, list):
        context = "\n".join(context)

    query = """Summarize the provided context and answer the following question as succintly as possible.
    Use ONLY the context provided below to complete this task.

    ```question
    {prompt}
    ```

    ```context
    {context}
    ```
    """.format(prompt=query, context=context)

    return retrieve_response(query, model='gpt-4o')


def summarize_thread(self, messages):
    """Summarizes the current conversation based on OpenAI chat messages."""

    # Assemble the conversation into a single string
    message = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

    # Create a new prompt for summarization
    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant that summarizes conversations as concisely as possible."},
        {"role": "user", "content": f"Summarize the following conversation briefly:\n\n{message}"}
    ]

    # Get summary
    return retrieve_response(summary_prompt)


def retrieve_response(self, prompt, model="gpt-4o-mini", temp=0.7, seed=42):
    """Get chat completion from ChatGPT"""

    completion = self.client.chat.completions.create(
        model=model,
        messages=summary_prompt,
        temperature=temp,
        seed=seed)

    return completion.choices[0].message['content'].strip()