import json
import os
import requests
from typing import Dict, Any, Union
import logging
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_with_llm(query: str, relevant_info: list) -> Union[str, Dict[str, Any]]:
    """
    Process the query and relevant information using an LLM to generate a structured output.

    Args:
        query (str): The original search query.
        relevant_info (list): A list of relevant information extracted from the search results.

    Returns:
        Union[str, Dict[str, Any]]: The structured output from the LLM or an error dictionary.
    """
    prompt = f"""
    You are an AI assistant tasked with analyzing and summarizing information about an entity.
    
    Query: {query}
    
    Relevant Information:
    {json.dumps(relevant_info, indent=2)}
    
    Based on the query and relevant information provided, create a comprehensive and accurate summary 
    of the entity in question. Your response should be structured as follows:

    ## Entity Summary

    ### Entity Name
    [Provide a clear and descriptive name for the entity]

    ### Entity Type
    [Specify the type of entity (e.g., person, organization, location, concept, etc.)]

    ### Key Facts
    - [Fact 1]
    - [Fact 2]
    - [Fact 3]
    - [Fact 4]
    - [Fact 5]

    ### Related Entities
    1. [Related Entity 1]: [Brief description of relationship]
    2. [Related Entity 2]: [Brief description of relationship]
    3. [Related Entity 3]: [Brief description of relationship]

    ### Sources
    - [Source 1](URL1)
    - [Source 2](URL2)
    - [Source 3](URL3)

    ### Confidence Level
    [Provide a confidence level (High, Medium, Low) for the information presented, based on the quality and consistency of the sources]

    ### Additional Notes
    [Include any caveats, potential biases, or areas where information might be incomplete or contradictory]

    Please ensure that your response is:
    1. Accurate and fact-based, drawing from the provided relevant information.
    2. Comprehensive, covering all aspects of the entity as available in the data.
    3. Objective, avoiding personal opinions or speculations.
    4. Properly formatted in Markdown for easy reading and parsing.
    5. Include clickable source links in the Sources section.
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,  # Set a timeout for the request
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        llm_response = response.json()

        if "response" in llm_response:
            return llm_response["response"]
        else:
            logger.error("LLM response does not contain 'response' key")
            return {
                "error": "Unexpected LLM response format",
                "details": "The 'response' key is missing from the LLM output.",
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Request to LLM failed: {str(e)}")
        return {"error": "Failed to communicate with the LLM", "details": str(e)}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
        return {"error": "Invalid JSON response from LLM", "details": str(e)}

    except Exception as e:
        logger.error(f"Unexpected error in process_with_llm: {str(e)}")
        return {"error": "An unexpected error occurred", "details": str(e)}
