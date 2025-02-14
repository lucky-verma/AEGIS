import requests
import logging

logger = logging.getLogger(__name__)


def is_ollama_running(ollama_url: str = "http://localhost:11434") -> bool:
    """
    Checks if Ollama is running.

    Returns:
    bool: True if Ollama is running, False otherwise.
    """
    try:
        response = requests.get(f"{ollama_url}/api/version")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def ensure_model_available(
    model_name: str, ollama_url: str = "http://localhost:11434"
) -> bool:
    """
    Checks if the specified Ollama model is available and pulls it if not.

    Args:
    model_name (str): The name of the Ollama model to check/pull.
    ollama_url (str): The base URL of the Ollama API. Defaults to "http://localhost:11434".

    Returns:
    bool: True if the model is available (either already or after pulling), False otherwise.
    """
    try:
        # Check if the model exists
        response = requests.get(f"{ollama_url}/api/show", params={"name": model_name})
        if response.status_code == 200:
            logger.info(f"Model {model_name} is already available.")
            return True
        elif response.status_code == 404:
            logger.info(f"Model {model_name} not found. Pulling it now...")
            # Pull the model
            pull_response = requests.post(
                f"{ollama_url}/api/pull", json={"name": model_name}
            )
            if pull_response.status_code == 200:
                logger.info(f"Model {model_name} successfully pulled.")
                return True
            else:
                logger.error(
                    f"Failed to pull model {model_name}. Status code: {pull_response.status_code}"
                )
                return False
        else:
            logger.warning(
                f"Unexpected response from Ollama API: {response.status_code}"
            )
            return False
    except requests.RequestException as e:
        logger.error(f"Error connecting to Ollama API: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking/pulling model: {str(e)}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ollama_url = "http://localhost:11434"  # Adjust this if your Docker setup uses a different URL

    if is_ollama_running(ollama_url):
        print("Ollama is running.")
        model_name = "nomic-embed-text"
        if ensure_model_available(model_name, ollama_url):
            print(f"Model {model_name} is ready to use.")
        else:
            print(f"Failed to ensure model {model_name} is available.")
    else:
        print("Ollama is not running.")
