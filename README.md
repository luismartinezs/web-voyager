# Web Voyager

Web Voyager is an AI agent that autonomously navigates and interacts with web pages using Playwright. It uses GPT-4 to interpret web content and make decisions on how to interact with the page to accomplish user-defined tasks.


![Screenshot 2024-11-24 at 09 19 32](https://github.com/user-attachments/assets/20d9d4f8-9d3c-423e-a95a-b8a4b21ff7de)

## Features

- Autonomous web navigation and interaction
- Visual element recognition and labeling
- Task-oriented decision making
- HTML report generation with screenshots

## Requirements

- Python 3.7+
- Playwright
- LangChain
- OpenAI API key

## Usage

1. Set up environment variables:
   - `LANGCHAIN_API_KEY`
   - `OPENAI_API_KEY`

2. Run the agent:

```python
from voyager import call_agent, async_playwright
import asyncio

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")
        result = await call_agent("Your task description here", page)
        print(f"Final response: {result}")
        await browser.close()

asyncio.run(main())
```

3. View the generated `web_voyager_results.html` for a step-by-step breakdown of the agent's actions and screenshots.

## How it Works

1. The agent takes a screenshot of the current page
2. It annotates interactive elements with bounding boxes
3. GPT-4 analyzes the page content and decides on the next action
4. The agent performs the action (click, type, scroll, etc.)
5. This process repeats until the task is completed

## Limitations

- Requires a valid OpenAI API key with GPT-4 access
- Performance may vary depending on the complexity of the web pages and tasks
