from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter


class PageUrl(BaseModel):
    url: HttpUrl


class WebScraperService:
    def __init__(self):
        self.crawler = AsyncWebCrawler(verbose=True, headless=True)
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

    async def start(self):
        await self.crawler.__aenter__()

    async def stop(self):
        await self.crawler.__aexit__(None, None, None)

    async def extract(self, url: str) -> str:
        result = await self.crawler.arun(url=url, config=self.default_config)

        if not result.success:
            raise HTTPException(status_code=400, detail="Failed to crawl the page")

        # Return the filtered markdown
        return result.markdown_v2.fit_markdown


app = FastAPI()
scraper_service = WebScraperService()


@app.on_event("startup")
async def startup_event():
    await scraper_service.start()


@app.on_event("shutdown")
async def shutdown_event():
    await scraper_service.stop()


@app.get("/")
async def root():
    return {"message": "Web Scraper Service is running"}


@app.post("/get-details")
async def get_url_data(page_url: PageUrl):
    try:
        page_data = await scraper_service.extract(str(page_url.url))
        return {"content": page_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
