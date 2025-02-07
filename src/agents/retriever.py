import httpx
import logging
from typing import List, Dict


class RetrieverAgent:
    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.logger = logging.getLogger(__name__)

    async def retrieve(self, query: str) -> List[Dict]:
        contexts = []
        current_query = query

        for hop in range(self.max_hops):
            try:
                # Get search results
                results = await self._search(current_query)

                # Process and validate results
                processed = await self._process_results(results)
                contexts.extend(processed)

                # Determine next query
                current_query = self._refine_query(query, contexts)

                if self._should_stop(contexts):
                    break

            except Exception as e:
                self.logger.error(f"Hop {hop+1} error: {str(e)}")

        return contexts

    async def _search(self, query: str) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://host.docker.internal:8080/search",
                params={
                    "q": f"{query}",
                    "format": "json",
                    "num_results": 10,
                },
            )
            return [
                r for r in response.json().get("results", []) if "umbc.edu" in r["url"]
            ]

    async def _process_results(self, results: List[Dict]) -> List[Dict]:
        processed = []
        async with httpx.AsyncClient() as client:
            for result in results:
                try:
                    response = await client.post(
                        "http://host.docker.internal:8081/crawl", json={"url": result["url"]}
                    )
                    content = response.json().get("content", "")
                    processed.append(
                        {
                            "title": result.get("title", ""),
                            "url": result["url"],
                            "content": content[:2000],
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Processing error: {str(e)}")
        return processed

    def _refine_query(self, original: str, contexts: List[Dict]) -> str:
        # Implement query refinement logic based on contexts
        return f"{original} - refined"

    def _should_stop(self, contexts: List[Dict]) -> bool:
        # Implement stopping condition
        return len(contexts) >= 5
