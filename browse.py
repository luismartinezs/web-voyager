from voyager import call_agent, async_playwright
import asyncio
import logging
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(filename='demo_errors.log', level=logging.ERROR)

# Add this global variable at the top of the file
is_processing_query = False

load_dotenv()

def get_profile_image_base64():
    try:
        image_path = Path(__file__).parent / 'me.jpeg'
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f'data:image/jpeg;base64,{encoded_string}'
    except Exception as e:
        logging.error(f"Error loading profile image: {str(e)}")
        return ''  # Return empty string if image loading fails

async def inject_chat_interface(page):
    profile_image = get_profile_image_base64()
    print(f"Profile image base64: {profile_image}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await page.evaluate(f"""
            () => {{
                if (document.getElementById('ai-chat-window')) return;

                const chatHtml = `
                    <div id="ai-chat-window" style="position: fixed; bottom: 20px; right: 20px; width: 300px; background-color: white; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); z-index: 2147483647; display: none; flex-direction: column; font-family: Arial, sans-serif;">
                        <div style="background-color: #4a90e2; color: white; padding: 10px; border-top-left-radius: 10px; border-top-right-radius: 10px; display: flex; align-items: center;">
                            <img src="{profile_image}" style="width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;">
                            <h3 style="margin: 0; color: white;">Elins Assistent</h3>
                        </div>
                        <div id="ai-chat-messages" style="flex: 1; overflow-y: auto; padding: 10px; max-height: 300px;"></div>
                        <div id="ai-chat-input-area" style="display: flex; padding: 10px; border-top: 1px solid #eee;">
                            <input id="ai-chat-input" type="text" style="flex: 1; margin-right: 10px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; outline: none;">
                            <button id="ai-chat-send" style="background-color: #4a90e2; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer;">Send</button>
                        </div>
                    </div>
                `;

                document.body.insertAdjacentHTML('beforeend', chatHtml);

                const chatWindow = document.getElementById('ai-chat-window');
                const chatMessages = document.getElementById('ai-chat-messages');
                const chatInput = document.getElementById('ai-chat-input');
                const chatSend = document.getElementById('ai-chat-send');
                const chatInputArea = document.getElementById('ai-chat-input-area');

                // Load previous messages from localStorage
                const savedMessages = localStorage.getItem('aiChatMessages');
                if (savedMessages) {{
                    chatMessages.innerHTML = savedMessages;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }}

                chatSend.addEventListener('click', () => {{
                    const query = chatInput.value;
                    if (query.trim()) {{
                        const userMessage = `<p><strong>You:</strong> ${{query}}</p>`;
                        chatMessages.innerHTML += userMessage;
                        chatInput.value = '';
                        chatInputArea.style.display = 'none';
                        chatMessages.innerHTML += '<p><em>Elins Agent working...</em></p>';
                        chatMessages.scrollTop = chatMessages.scrollHeight;

                        // Save messages to localStorage
                        localStorage.setItem('aiChatMessages', chatMessages.innerHTML);

                        window.callAgent(query);
                    }}
                }});

                chatInput.addEventListener('keypress', (e) => {{
                    if (e.key === 'Enter') {{
                        chatSend.click();
                    }}
                }});

                window.addAgentResponse = (response) => {{
                    chatMessages.innerHTML = chatMessages.innerHTML.replace('<p><em>Elins Agent working...</em></p>', '');
                    const agentMessage = `<p><strong>Agent:</strong> ${{response}}</p>`;
                    chatMessages.innerHTML += agentMessage;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    chatInputArea.style.display = 'flex';

                    // Save messages to localStorage
                    localStorage.setItem('aiChatMessages', chatMessages.innerHTML);

                    // Signal that processing is complete
                    window.isProcessingQuery = false;
                }};

                // Add this line to initialize the processing state
                window.isProcessingQuery = false;

                // Initialize chat window with display:none, then show after a brief delay
                setTimeout(() => {{
                    chatWindow.style.display = 'flex';
                }}, 500);
            }}
            """)
            break  # If successful, exit the loop
        except Exception as e:
            if attempt == max_retries - 1:  # If this was the last attempt
                logging.error(f"Failed to inject chat interface after {max_retries} attempts: {str(e)}")
            else:
                await asyncio.sleep(1)  # Wait a bit before retrying

async def call_agent_wrapper(query, page):
    global is_processing_query
    try:
        is_processing_query = True
        await page.evaluate("window.isProcessingQuery = true;")

        result = await call_agent(query, page)
        print(f"Final response: {result}")

        # Verify page is still valid before evaluating
        if not page.is_closed():
            await page.evaluate(f"window.addAgentResponse({json.dumps(result)})")
    except Exception as e:
        logging.error(f"Error in call_agent_wrapper: {str(e)}")
        if not page.is_closed():
            await page.evaluate("window.addAgentResponse('Sorry, an error occurred while processing your request.')")
    finally:
        is_processing_query = False

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Set up event listener for page creation
        context.on("page", lambda page: asyncio.ensure_future(setup_new_page(page)))

        page = await context.new_page()
        await page.goto("https://www.google.com")

        # Set up the initial page
        await setup_page(page)

        print("Browser is open. Use the chat window to interact with the agent.")
        print("Close the browser window to exit.")

        # Keep the script running until interrupted
        while True:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break

    print("Browser has been closed. Exiting the program.")

async def setup_new_page(page):
    await page.wait_for_load_state('domcontentloaded')
    await setup_page(page)

async def setup_page(page):
    try:
        await inject_chat_interface(page)
        await page.expose_function("callAgent", lambda query: asyncio.ensure_future(call_agent_wrapper(query, page)))

        # Modified navigation handler with error handling
        async def on_navigation():
            try:
                global is_processing_query
                await inject_chat_interface(page)
                await asyncio.sleep(0.5)

                if not page.is_closed():  # Check if page is still valid
                    if not is_processing_query:
                        await page.evaluate("document.getElementById('ai-chat-window').style.display = 'flex';")
                    else:
                        await page.evaluate("document.getElementById('ai-chat-input-area').style.display = 'none';")
            except Exception as e:
                logging.error(f"Error in on_navigation: {str(e)}")

        page.on("load", lambda _: asyncio.ensure_future(on_navigation()))
    except Exception as e:
        logging.error(f"Error in setup_page: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error("Error in main: %s", str(e))