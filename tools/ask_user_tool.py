# tools/ask_user_tool.py
from langchain_core.tools import tool
from langgraph.types import interrupt

@tool
def ask_user(question: str) -> str:
    '''
    Ask the user a question and wait for their response.
    Pauses execution until user provides input.
    '''
    # This actually pauses the graph and waits!
    response: str = interrupt(question)
    return response
