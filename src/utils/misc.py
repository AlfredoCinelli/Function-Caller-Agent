"""Module aimed at creating miscellaneous functions."""

# Import  packages and modules
from typing import Union

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool

# Define functions


def find_tool_by_name(
    tools: list[Tool],
    tool_name: str,
) -> Tool:

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
