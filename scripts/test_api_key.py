"""
Test API Key Script

This script tests the functionality of the OpenAI API using a provided API key.
It sends a simple chat message to the Deepseek model and prints the response.

Note: Always keep API keys secure and never commit them to version control.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with the API key and base URL
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Get API key from environment variables
    base_url=os.getenv("OPENAI_BASE_URL")  # Get base URL from environment variables
)

# Create a chat completion request
response = client.chat.completions.create(
    model="deepseek-chat",  # Specify the model to use
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},  # System message
        {"role": "user", "content": "Hello"},  # User message
    ],
    stream=False,  # Disable streaming for this simple test
)

# Print the assistant's response
print(response.choices[0].message.content)
