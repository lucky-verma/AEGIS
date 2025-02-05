import requests
import time
from requests.exceptions import RequestException
from dotenv import load_dotenv
import os
import aiohttp
import asyncio

load_dotenv()


class SearXNGSearch:
    def __init__(self, base_url=None, max_retries=3, backoff_factor=1):
        self.base_url = base_url or os.getenv("SEARXNG_URL", "http://localhost:8080")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()

    def search(self, query, num_results=10):
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


searx = SearXNGSearch()


async def scrape_url(session, url):
    try:
        async with session.post(
            "http://localhost:8081/get-details", json={"url": url}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("content", "")
            else:
                print(f"Error scraping {url}: {response.status}")
                return ""
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""


async def perform_search_async(query, num_results=20):
    searxng_results = searx.search(query, num_results)
    searxng_urls = [result["url"] for result in searxng_results]

    async with aiohttp.ClientSession() as session:
        tasks = [scrape_url(session, url) for url in searxng_urls]
        crawl4ai_contents = await asyncio.gather(*tasks)

    results = []
    for searxng_result, crawl4ai_content in zip(searxng_results, crawl4ai_contents):
        results.append({"searxng": searxng_result, "crawl4ai": crawl4ai_content})

    print("Search results:")
    print(results[0])
    return results


def perform_search(query, num_results=20):
    return asyncio.run(perform_search_async(query, num_results))


if __name__ == "__main__":
    test_query = "Python programming"
    results = perform_search(test_query)
    for result in results[:2]:
        print(result["url"])
        print(result["content"][:200])
        print("---" * 10)
