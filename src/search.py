import requests
import time
from requests.exceptions import RequestException
from dotenv import load_dotenv
import os

load_dotenv()


class SearXNGSearch:
    def __init__(self, base_url=None, max_retries=3, backoff_factor=1):
        self.base_url = base_url or os.getenv("SEARXNG_URL", "http://localhost:8080")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()

    def search(self, query, num_results=20):
        params = {
            "q": query,
            "format": "json",
            "num_results": num_results,
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                results = response.json()
                return results.get("results", [])
            except RequestException as e:
                if response.status_code == 429:
                    wait_time = self.backoff_factor * (2**attempt)
                    print(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error occurred: {e}")
                    if attempt == self.max_retries - 1:
                        return []

        return []


# Initialize the SearXNGSearch instance
searx = SearXNGSearch()


def perform_search(query):
    """
    Perform a search using SearXNG and return the results.
    """
    results = searx.search(query)
    return results


# You can keep this for testing purposes
if __name__ == "__main__":
    test_query = "Python programming"
    results = perform_search(test_query)
    for result in results[:5]:  # Print top 5 results
        print(f"Title: {result.get('title', '')}")
        print(f"URL: {result.get('url', '')}")
        print("---")
