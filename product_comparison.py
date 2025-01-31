from crewai import Agent, Task
from pydantic import BaseModel, Field

class HtmlReport(BaseModel):
    html: str = Field(..., description="A html tablular format which can be easily displayed in a gradio interface.")

comparision_agent = Agent(
    role="Comparision Agent",
    goal="Compare two products: {product1_name} and {product2_name}",
    backstory="You are a comparision expert who specializes in comparing the two products: {product1_name} and {product2_name}",
    verbose=True,
)

comparision_task = Task(
    description="Compare the two products: {product1_name} and {product2_name} and create a report which contains exactly {num_highlights} comparisons. Details are as follows: {product1_name}:{product1_highlights} and {product2_name}:{product2_highlights}",
    expected_output="A html tablular format which can be easily displayed in a gradio interface.",
    agent=comparision_agent,
    output_pydantic=HtmlReport
)
