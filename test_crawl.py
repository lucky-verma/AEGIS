import asyncio
import nest_asyncio
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
)

nest_asyncio.apply()


async def simple_crawl():
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.kidocode.com/degrees/technology", config=crawler_run_config
        )
        print(
            result.markdown_v2.raw_markdown[:500].replace("\n", " -- ")
        )  # Print the first 500 characters


asyncio.run(simple_crawl())
