import logging
import os # Used for file path handling

# --- Basic Configuration Example ---
# This is often sufficient for simple scripts.
# If basicConfig is called, it configures the root logger.
# It should ideally be called only once, early in your script.
# NOTE: If any logging happens before basicConfig, it might not work as expected.

print("--- Running Basic Logging Example ---")
logging.basicConfig(
    level=logging.INFO, # Set the minimum level of messages to handle (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s | %(levelname)-8s | %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the date/time format
    # filename='basic_app.log', # Uncomment this to log to a file instead of the console
    # filemode='w' # Use 'w' to overwrite the file, 'a' to append (default)
)

# Now we can log messages using the root logger directly
logging.debug("This is a debug message (won't be shown with INFO level)")
logging.info("Application starting up.")
logging.warning("Something might be configured incorrectly.")
logging.error("An error occurred processing a request.")
logging.critical("Critical failure! Application shutting down.")
print("-" * 40)


# --- Module-Level Logger with Handlers Example ---
# This is a more robust pattern, especially for larger applications or libraries.

print("\n--- Running Handler-Based Logging Example ---")

# 1. Get a specific logger instance (best practice: use the module name)
# Using __name__ ensures the logger name reflects the module hierarchy.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set the logger's threshold - it will process DEBUG and higher.

# Prevent propagation if basicConfig was already called and you want isolation
# logger.propagate = False # Uncomment if you don't want messages going to root logger handlers

# 2. Create Handlers (where the logs go)

# Console Handler (StreamHandler): Outputs to stderr/stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Console shows INFO level and above

# File Handler: Outputs to a file
log_file = 'detailed_app.log'
# Ensure the log file is cleared at the start for this example run
if os.path.exists(log_file):
    os.remove(log_file)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG) # File captures everything (DEBUG and above)

# 3. Create a Formatter (how the logs look)
formatter = logging.Formatter(
    '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 4. Set the Formatter for each Handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 5. Add the Handlers to the Logger
# Avoid adding handlers multiple times if this code runs more than once
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
else:
    # If re-running in an interactive session, clear existing handlers first
    # Note: In a typical script, you wouldn't need this 'else' block
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# 6. Example Usage within a function/module
def process_data(data):
    logger.info(f"Starting to process data: {data}")
    try:
        logger.debug(f"Data type: {type(data)}")
        if not isinstance(data, str):
            logger.warning("Input data is not a string, attempting conversion.")
            data = str(data)

        result = data.upper()
        logger.debug(f"Processing successful, result: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to process data: {data}", exc_info=True)
        # Use logger.exception("...") as a shorthand for logger.error("...", exc_info=True)
        # logger.exception(f"Failed to process data: {data}")
        return None

# Run the function
process_data("Sample Data")
process_data(123) # Example that triggers a warning
process_data([1, 2]) # Example that might cause an error depending on implementation detail

logger.info("Finished processing.")
print("-" * 40)

print(f"\nCheck the console output above (should show INFO, WARNING, ERROR).")
print(f"Check the file '{log_file}' for detailed logs (including DEBUG messages).")

# Example content of detailed_app.log:
# 2025-04-22 06:16:27 | __main__     | INFO     | Starting to process data: Sample Data
# 2025-04-22 06:16:27 | __main__     | DEBUG    | Data type: <class 'str'>
# 2025-04-22 06:16:27 | __main__     | DEBUG    | Processing successful, result: SAMPLE DATA
# 2025-04-22 06:16:27 | __main__     | INFO     | Starting to process data: 123
# 2025-04-22 06:16:27 | __main__     | DEBUG    | Data type: <class 'int'>
# 2025-04-22 06:16:27 | __main__     | WARNING  | Input data is not a string, attempting conversion.
# 2025-04-22 06:16:27 | __main__     | DEBUG    | Processing successful, result: 123
# 2025-04-22 06:16:27 | __main__     | INFO     | Starting to process data: [1, 2]
# 2025-04-22 06:16:27 | __main__     | DEBUG    | Data type: <class 'list'>
# 2025-04-22 06:16:27 | __main__     | WARNING  | Input data is not a string, attempting conversion.
# 2025-04-22 06:16:27 | __main__     | DEBUG    | Processing successful, result: [1, 2]
# 2025-04-22 06:16:27 | __main__     | INFO     | Finished processing.

