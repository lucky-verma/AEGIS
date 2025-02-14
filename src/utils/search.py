import httpx
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SearxNGWrapper:
    def __init__(self):
        self.base_url = "http://localhost:8080"

    async def search(
        self, query: str, num_results: int = 3, engines: List[str] = None
    ) -> List[Dict]:
        try:
            params = {
                "q": query,
                "format": "json",
                "language": "en",
                "max_results": num_results,
            }

            if engines:
                params["engines"] = ",".join(engines)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])[:num_results]
                    return results
                else:
                    logger.error(f"Search Error: {response.text}")
                    return []
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
