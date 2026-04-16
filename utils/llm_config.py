# utils/llm_config.py
# This file creates the LLM (AI brain) that all agents will share.
# Instead of repeating this setup in every agent file, we write it
# once here and simply import it wherever we need it.

import os
from groq import Groq
from dotenv import load_dotenv

# Load the API key from our .env file
load_dotenv()

# This function returns a configured Groq client
# Any agent can call this function to get the AI brain
def get_llm_client():
    client = Groq(
        api_key=os.getenv("gsk_lA911g5FZn2yTKShw7kSWGdyb3FYJOC1XS6wMAbGaNoAD3nyzCW")
    )
    return client

# The model name we use across the entire project
# Written here once so we never have to remember it elsewhere
MODEL_NAME = "llama-3.1-8b-instant"