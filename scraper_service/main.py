from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import uvicorn
import nest_asyncio

# Apply nest_asyncio for async context compatibility
nest_asyncio.apply()


class PageUrl(BaseModel):
    url: HttpUrl


class WebScraperService:
    def __init__(self):
        self.crawler = None
        self.default_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.6, min_word_threshold=20, threshold_type="dynamic"
                ),
                options={
                    "ignore_links": False,
                    "ignore_images": True,
                    "escape_html": False,
                },
            ),
            cache_mode=CacheMode.BYPASS,
        )

    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(verbose=True, headless=True)
        await self.crawler.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)

    async def extract(self, url: str) -> str:
        result = await self.crawler.arun(url=url, config=self.default_config)
        if not result.success or len(result.markdown_v2.fit_markdown) < 100:
            raise HTTPException(status_code=400, detail="Insufficient content")
        return result.markdown_v2.fit_markdown


app = FastAPI()
scraper_service = WebScraperService()


@app.on_event("startup")
async def startup_event():
    await scraper_service.__aenter__()


@app.on_event("shutdown")
async def shutdown_event():
    await scraper_service.__aexit__(None, None, None)


@app.get("/")
async def root():
    return {"status": "ready"}


@app.post("/crawl")
async def crawl_page(page_url: PageUrl):
    try:
        content = await scraper_service.extract(str(page_url.url))
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
