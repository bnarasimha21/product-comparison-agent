# Product Comparison Agent

An AI-powered agent that compares two products side-by-side by automatically searching the web, scraping product details, and generating a structured HTML comparison report. Built with CrewAI for multi-agent orchestration and Gradio for the user interface.

## Features

- **Multi-agent pipeline** using CrewAI with three specialized agents:
  - **URL Finder Agent** -- Searches the web for relevant product pages using Serper
  - **Web Scraper Agent** -- Extracts key product highlights from discovered URLs
  - **Comparison Agent** -- Generates a structured HTML comparison table
- **CrewAI Flow orchestration** for sequential, event-driven agent execution
- **Gradio web interface** with configurable number of URLs and comparison highlights
- **Structured output** via Pydantic models ensuring consistent, parseable results
- **HTML tabular report** displayed directly in the browser

## Tech Stack

- **Language:** Python
- **Agent Framework:** CrewAI (with Flows)
- **Web UI:** Gradio
- **Web Search:** SerperDevTool (Google Search API)
- **Web Scraping:** ScrapeWebsiteTool (CrewAI Tools)
- **Data Validation:** Pydantic
- **Environment Management:** python-dotenv

## Setup / Installation

Requires Python 3.10 or newer (current CrewAI releases do not support older versions).

```bash
git clone https://github.com/bnarasimha21/product-comparison-agent.git
cd product-comparison-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys (`OPENAI_API_KEY` drives the agents' default LLM, `SERPER_API_KEY` is used by the search tool):

```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

## Usage

```bash
python main.py
```

1. Open the Gradio interface in your browser (default: `http://localhost:7860`)
2. Enter two product names to compare (e.g., "iPhone 15 Pro" and "Samsung Galaxy S24 Ultra")
3. Select the number of URLs to search per product (1-5)
4. Select the number of highlights to compare (1-10)
5. Click **Submit** and wait for the agents to research and generate the comparison report

The output is an HTML table comparing the two products across the selected number of feature highlights.

## License

MIT
