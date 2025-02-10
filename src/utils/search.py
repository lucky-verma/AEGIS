import httpx
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SearxNGWrapper:
    def __init__(self):
        self.base_url = "http://localhost:8080"

    async def search(self, query: str, num_results: int = 3) -> List[Dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query, "format": "json", "num_results": num_results},
                )
                print(f"Response status code: {response.status_code} | URL: {response.url}")
                return response.json().get("results", [])
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
