from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List
from crewai_tools import SerperDevTool

class ProductDetails(BaseModel):
    name: str = Field(..., description="The full name of the product.")
    urls: List[str] = Field(..., description="The urls where we can find more details about the product.")

    
product_details_finder = Agent(
    role="Url Finder",
    goal="Find the website Url and other Urls where we can find more details about: {product_name}",
    backstory="You are a Url finder who specializes in finding the correct Urls for: {product_name}", 
    verbose=True,
)

product_details_task = Task(
    description="Find the Urls for {product_name}",
    expected_output="Product name and Urls of {product_name}. Include only {num_urls} Urls for each product",
    agent=product_details_finder,
    tools=[SerperDevTool(query="{product_name}")],
    output_pydantic=ProductDetails
)