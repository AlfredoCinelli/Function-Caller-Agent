"""Module aimed at creating miscellaneous functions."""

# Import  packages and modules
from typing import Union

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain_core.documents import Document
import json
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import END

# Define functions


def find_tool_by_name(
    tools: list[Tool],
    tool_name: str,
) -> Tool:

    tool_name = tool_name.split("(")[0] if "(" in tool_name else tool_name
    tool = next(
        (tool for tool in tools if tool.name == tool_name),
        None,
    )
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")
    return tool


class RobustReActOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        else:
            # Extract Action and Action Input even if format isn't perfect
            action_match = text.split("Action:")[-1].split("Action Input:")[0].strip()
            action_input_match = text.split("Action Input:")[-1].strip()

            return AgentAction(
                tool=action_match, tool_input=action_input_match, log=text
            )

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    
def chatbot(state: State, llm):
    return {"messages": [llm.invoke(state["messages"])]}

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

#def chatbot(state: State, llm) -> dict:
#    system_prompt = """
#    You are an AI assistant.
#    Answer the following questions as best you can.
#    You have access to tools, call them only if necessary.
#    
#    Use the following format:
#
#    Question: the input question you must answer
#    Thought: you should always think about what to do
#    Action: the action to take
#    Action Input: the input to the action
#    Observation: the result of the action
#    ... (this Thought/Action/Action Input/Observation can repeat N times)
#    Thought: whether you know the final answer or not
#    Final Answer: the final answer to the original input question
#
#    Begin!
#    """
#    return {"messages": [llm.invoke([system_prompt] + state["messages"])]}