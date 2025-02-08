from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import uvicorn
import tempfile
import aiohttp
import os
from markitdown import MarkItDown
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PageUrl(BaseModel):
    url: HttpUrl


class EnhancedWebScraperService:
    def __init__(self):
        self.crawler = None
        self.markdown_converter = MarkItDown()
        self.supported_mimetypes = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.ms-excel": "xls",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        }
        self.default_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.2, min_word_threshold=10, threshold_type="dynamic"
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

    async def detect_content_type(
        self, url: str
    ) -> Tuple[Optional[str], Optional[bytes]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(url) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if any(mime in content_type for mime in self.supported_mimetypes):
                        async with session.get(url) as download:
                            content = await download.read()
                            return content_type.split(";")[0], content
            return None, None
        except Exception as e:
            logger.error(f"Error detecting content type: {str(e)}")
            return None, None

    async def convert_file_to_markdown(self, content: bytes, content_type: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{self.supported_mimetypes[content_type]}"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                result = self.markdown_converter.convert(tmp_path)
                return result.text_content
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error converting file to markdown: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"File conversion error: {str(e)}"
            )

    async def extract(self, url: str) -> str:
        content_type, content = await self.detect_content_type(url)

        if content_type and content:
            logger.info(f"Converting file of type {content_type} to markdown")
            return await self.convert_file_to_markdown(content, content_type)

        # Fall back to regular web crawling for HTML content
        try:
            result = await self.crawler.arun(url=url, config=self.default_config)
            if not result.success or len(result.markdown_v2.fit_markdown) < 100:
                raise HTTPException(status_code=400, detail="Insufficient content")
            return result.markdown_v2.fit_markdown
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


app = FastAPI()
scraper_service = EnhancedWebScraperService()


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
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
