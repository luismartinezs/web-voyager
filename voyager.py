# Optional: add tracing to visualize the agent trajectories
import os
from getpass import getpass
import warnings
import sys
import asyncio
from dotenv import load_dotenv

if sys.platform == 'darwin':
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

def _getpass(env_var: str):
    # Check if variable exists in environment (including .env file)
    if not os.getenv(env_var):
        os.environ[env_var] = getpass(f"{env_var}=")

# Add this before setting environment variables
load_dotenv()  # This will load environment variables from .env file

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager"
_getpass("LANGCHAIN_API_KEY")
_getpass("OPENAI_API_KEY")


from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page



class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool



import platform

# TOOLS

async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]

    # Check if the element is visible and clickable
    is_visible = await page.evaluate(f"""
        () => {{
            const element = document.elementFromPoint({x}, {y});
            if (!element) return false;
            const rect = element.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 &&
                   window.getComputedStyle(element).visibility !== 'hidden';
        }}
    """)

    if not is_visible:
        return f"Element at bbox {bbox_id} is not visible or clickable"

    await page.mouse.click(x, y)
    await asyncio.sleep(2)

    # Check if the page URL changed after clicking
    new_url = page.url

    return f"Clicked {bbox_id}. Element type: {bbox['type']}. Text: '{bbox['text']}'. URL after click: {new_url}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    await asyncio.sleep(2)
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."


import base64
from PIL import Image
import io
from langchain_core.runnables import chain as chain_decorator

# Some javascript we will run on each step
# to take a screenshot of the page, select the
# elements to annotate, and add bounding boxes
with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("""
                () => {
                    const bboxes = markPage();
                    return bboxes.map(bbox => {
                        const element = document.elementFromPoint(bbox.x, bbox.y);
                        bbox.zIndex = element ? window.getComputedStyle(element).zIndex : 'auto';
                        return bbox;
                    });
                }
            """)
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await page.screenshot()


    image = Image.open(io.BytesIO(screenshot))
    resized_image = image.resize((800, 600))

    with io.BytesIO() as output:
        resized_image.save(output, format="PNG")
        resized_screenshot = output.getvalue()

    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(resized_screenshot).decode(),
        "bboxes": bboxes,
    }


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        z_index = bbox.get("zIndex", "unknown")
        labels.append(f'{i} (<{el_type}/> z-index: {z_index}): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


# Will need a later version of langchain to pull
# this image prompt template
prompt = hub.pull("samthesquirrel/web-voyager")

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)


import re


def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        # Find all step numbers in the text
        steps = re.findall(r'\n(\d+)\. ', txt)
        if steps:
            # Get the last (highest) step number
            step = int(steps[-1]) + 1
        else:
            step = 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}



from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, StateGraph

graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}


for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation"
        # key in the AgentState
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")


def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action


graph_builder.add_conditional_edges("agent", select_tool)


graph = graph_builder.compile()


# from IPython import display
from playwright.async_api import async_playwright

import base64


class IncrementalHTMLGenerator:
    def __init__(self):
        self.steps = []
        self.html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Voyager Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .step {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .step-content {
            white-space: pre-wrap;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Web Voyager Results</h1>
    <div id="results">
"""

    def add_step(self, action, action_input, img_data):
        step_number = len(self.steps) + 1
        step_content = f"{step_number}. {action}: {action_input}"
        self.steps.append(step_content)

        self.html_content += f"""
        <div class="step">
            <div class="step-content">{step_content}</div>
            <img src="data:image/png;base64,{img_data}" alt="Step {step_number} Image">
        </div>
"""

    def set_final_answer(self, final_answer):
        self.html_content += f"""
    </div>
    <div id="final-answer">
        <h2>Final Answer</h2>
        <p>{final_answer}</p>
    </div>
</body>
</html>
"""

    def write_html(self, filename):
        with open(filename, 'w') as f:
            f.write(self.html_content)

async def call_agent(question: str, page, max_steps: int = 150):
    print(f"Calling agent with question: {question}")
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    html_generator = IncrementalHTMLGenerator()

    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")

        html_generator.add_step(action, action_input, event["agent"]["img"])

        if "ANSWER" in action:
            final_answer = action_input[0]
            break

    html_generator.set_final_answer(final_answer)
    html_generator.write_html("web_voyager_results.html")

    return final_answer

def generate_html(steps, images, final_answer):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    ... (HTML head and style from the artifact) ...
    <body>
        <h1>Web Voyager Results</h1>
        <div id="results">
            {steps_html}
        </div>
        <div id="final-answer">
            <h2>Final Answer</h2>
            <p>{final_answer}</p>
        </div>
    </body>
    </html>
    """

    steps_html = ""
    for i, (step, img) in enumerate(zip(steps, images)):
        steps_html += f"""
        <div class="step">
            <div class="step-content">{i+1}. {step}</div>
            <img src="data:image/png;base64,{img}" alt="Step {i+1} Image">
        </div>
        """

    return html_template.format(steps_html=steps_html, final_answer=final_answer)


warnings.filterwarnings("ignore")



