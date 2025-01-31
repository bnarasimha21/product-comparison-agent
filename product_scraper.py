from crewai import Agent, Task
from crewai_tools import ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List

class ProductHighlights(BaseModel):
    name: str = Field(..., description="The name of the product")
    highlights: List[str] = Field(..., description="A list of top {num_highlights} highlights about given product {product_name}")

product_scraper_agent = Agent(
    role="Web Scraper",
    goal="Scrape the given Urls: {urls} for {product_name}",
    backstory="You are a web scraper who specializes in scraping the given Urls: {urls} for {product_name}",
    verbose=True
)

product_scrape_task = Task(
    description="Scrape the given Urls: {urls} for {product_name}",
    expected_output="A list of top {num_highlights} highlights about given product {product_name}",
    agent=product_scraper_agent,
    tools=[ScrapeWebsiteTool(urls="{urls}")],
    output_pydantic=ProductHighlights
)

