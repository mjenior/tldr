

from openai import OpenAI
from datetime import datetime

from .completion import summarize_thread

def start_new_thread(self, context=False, message_limit=20):
    """Start a new thread with only the current agent and adds previous context if needed."""
    
    # Initialize thread
    global thread
    thread = self.beta.threads.create()
    self.current_thread = thread.id
    self.thread_history[self.current_thread.id] = str(datetime.now())
    
    # Apply new attributes
    self.current_thread.message_limit = message_limit
	self.current_thread.current_thread_calls = 0
	self.current_thread.current_cost = 0.0
	self.current_thread.current_tokens = {}

	# Update client thread attributes
	
	
    # Add previous context
    if context == True:
    	conversation = summarize_thread(self.current_thread.messages)
        previous_context = client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=conversation)


# Initialize OpenAI client and conversation thread
def initialize_session(self, api_key=None):

	# Confirm environment API key
	if api_key is None:
		try:
			api_key = os.getenv("OPENAI_API_KEY")
		except EnvironmentError:
	    	raise EnvironmentError("OPENAI_API_KEY environment variable not found!")

	# Start client and first thread
	client = OpenAI(api_key=api_key)
	client.current_cost = 0.0
	client.current_tokens = {}
	client.thread_history = {}

	# Start initial thread
	client.start_new_thread = start_new_thread.__get__(client)
	client.start_new_thread()
	self.client = client
	

def summarize_thread(messages, model="gpt-4o-mini"):
    """Summarizes the current conversation based on OpenAI chat messages."""

    # Assemble the conversation into a single string
    message = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

    # Create a new prompt for summarization
    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant that summarizes conversations as concisely as possible."},
        {"role": "user", "content": f"Summarize the following conversation briefly:\n\n{message}"}]

    # Get the completion
    response = openai.ChatCompletion.create(
        model=model,
        messages=summary_prompt,
        temperature=0.7)

    return response.choices[0].message['content'].strip()
