import httpx
import logging
from typing import List, Dict
from utils.search import SearxNGWrapper


class RetrieverAgent:
    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.searxng = SearxNGWrapper()
        self.logger = logging.getLogger(__name__)

    async def retrieve(self, query: str) -> List[Dict]:
        contexts = []
        current_query = query

        for hop in range(self.max_hops):
            try:
                results = await self.searxng.search(current_query)
                processed = await self._process_results(results)
                contexts.extend(processed)

                if self._should_stop(contexts):
                    break

                current_query = self._refine_query(query, contexts)

            except Exception as e:
                self.logger.error(f"Hop {hop+1} error: {str(e)}")

        return contexts

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
        # Implement query refinement logic
        return f"{original} - refined"

    def _should_stop(self, contexts: List[Dict]) -> bool:
        return len(contexts) >= 10  # Stop after 10 contexts
