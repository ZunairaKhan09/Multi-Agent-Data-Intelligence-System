

from groq import Groq
from dotenv import load_dotenv
import os

# Load the API key from .env file
load_dotenv()

# Create connection to Groq
client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

# Send a message to the AI and get a reply
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Say hello and introduce yourself in 2 sentences."
        }
    ]
)

# Extract the text reply
reply = response.choices[0].message.content

print("AI Response:")
print(reply)