import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def process_with_llm(query, relevant_info):
    prompt = f"""
    Query: {query}
    Relevant Information: {json.dumps(relevant_info, indent=2)}
    ---
    Based on the query and relevant information provided, create a structured output with the following fields:

    1. **Entity Name**: a descriptive name for the entity
    2. **Entity Type**: the type of entity (e.g. person, organization, location, etc.)
    3. **Key Facts**: a list of 3-5 important facts about the entity
    4. **Related Entities**: a list of 2-3 related entities, if applicable
    5. **Sources**: a list of unique sources used to generate the output
    
    Please provide the output in markdown format, with each field separated by a blank line.
    """

    response = requests.post(
        OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )

    if response.status_code == 200:
        llm_response = response.json()
        try:
            if "response" in llm_response:
                return llm_response["response"]
            else:
                return {
                    "error": "LLM response does not contain 'response' key",
                    "response": llm_response,
                }
        except Exception as e:
            return {
                "error": f"Error processing LLM response: {str(e)}",
                "response": llm_response,
            }

    else:
        return {
            "error": f"LLM request failed with status code {response.status_code}",
            "response": response.text,
        }
