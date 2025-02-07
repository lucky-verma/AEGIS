import httpx


class SearxNGWrapper:
    def __init__(self):
        self.base_url = "http://host.docker.internal:8080"

    async def search(self, query: str, num_results: int = 5) -> list:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "num_results": num_results},
            )
            return response.json().get("results", [])
