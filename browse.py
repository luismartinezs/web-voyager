from voyager import call_agent, async_playwright
import asyncio
import logging

logging.basicConfig(filename='demo_errors.log', level=logging.ERROR)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")

        res = await call_agent("Could you check how long it takes to drive between Gothenburg and Skara?", page)
        print(f"Final response: {res}")

        await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error("Error in main: %s", str(e))
         