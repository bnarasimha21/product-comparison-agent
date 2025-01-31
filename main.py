import os
from dotenv import load_dotenv
import gradio as gr
from crewai import Crew,  Flow
from crewai.flow.flow import listen, start
from product_details import product_details_finder, product_details_task
from product_scraper import product_scraper_agent, product_scrape_task
from product_comparison import comparision_agent, comparision_task

load_dotenv()

class ProductComparisonFlow(Flow):
    @start()
    def find_product_details(self, progress=gr.Progress(track_tqdm=True)):  
        product_details_finding_crew = Crew(
            agents=[product_details_finder], 
            tasks=[product_details_task]
        )

        product1 = self.state["product1"]
        product2 = self.state["product2"]
        num_urls = self.state["num_urls"]

        product1_details = product_details_finding_crew.kickoff(inputs={"product_name": product1, "num_urls": num_urls})
        progress(0.1, desc="Found the relevant URLs for product 1...")
        
        product2_details = product_details_finding_crew.kickoff(inputs={"product_name": product2, "num_urls": num_urls})
        progress(0.2, desc="Found the relevant URLs for product 2...")
        return [product1_details, product2_details]
    
    @listen(find_product_details)
    def scrape_product_details(self, product_details_list, progress=gr.Progress(track_tqdm=True)):
        product_scraping_crew = Crew(
            agents=[product_scraper_agent], 
            tasks=[product_scrape_task]
        )

        product1_details, product2_details = product_details_list
            
        product1_urls = ", ".join(product1_details["urls"])
        product2_urls = ", ".join(product2_details["urls"])
        num_highlights = self.state["num_highlights"]

        product1_scrape = product_scraping_crew.kickoff(inputs={"product_name": product1_details["name"], "urls": product1_urls, "num_highlights": num_highlights})
        progress(0.4, desc="Collected product 1 details...")

        product2_scrape = product_scraping_crew.kickoff(inputs={"product_name": product2_details["name"], "urls": product2_urls, "num_highlights": num_highlights})
        progress(0.7, desc="Collected product 2 details...")

        return [product1_scrape, product2_scrape]

    @listen(scrape_product_details)
    def report_product_details(self, product_scrape_list, progress=gr.Progress(track_tqdm=True)):
        comparision_crew = Crew(
            agents=[comparision_agent], 
            tasks=[comparision_task]
        )

        product1_scrape, product2_scrape = product_scrape_list
        
        product1_name = product1_scrape["name"]
        product2_name = product2_scrape["name"]
        product1_highlights = product1_scrape["highlights"]
        product2_highlights = product2_scrape["highlights"]
        num_highlights = self.state["num_highlights"]

        progress(0.7, desc="Generating the report...")
        product_comparison_report = comparision_crew.kickoff(inputs={"product1_name": product1_name, "product1_highlights": product1_highlights, "product2_name": product2_name, "product2_highlights": product2_highlights, "num_highlights": num_highlights})
        return product_comparison_report

def product_comparison(product_name1, product_name2, num_urls, num_highlights):
    flow = ProductComparisonFlow()
    flow.state["product1"] = product_name1
    flow.state["product2"] = product_name2
    flow.state["num_urls"] = num_urls
    flow.state["num_highlights"] = num_highlights

    result = flow.kickoff()  

    return result["html"]

iface = gr.Interface(
    fn=product_comparison,
    inputs=[
        gr.Textbox(label="Product Name 1"), 
        gr.Textbox(label="Product Name 2"),
        gr.Dropdown(label="Number of URLs to Search", choices=["1", "2", "3", "4", "5"], value="2"),
        gr.Dropdown(label="Number of Highlights to compare", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], value="2")
        ],
    outputs=gr.HTML(label="Product Comparison Report"),
    title="Product Comparison",
    description="Enter two product names to compare their details."
)

iface.launch()








