"""Main script to assemble LangGraph Agent Graph."""

# Import packages and modules

import warnings
from functools import partial
from typing import Union

import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import Self

from src.utils.callbacks import LLMCallbackHandler
from src.utils.misc import (  # equivalent to ToolNode from langgraph.prebuilt; equivalent to tools_condition from langgraph.prebuilt
    BasicToolNode,
    State,
    chatbot,
    route_tools,
)
from src.utils.tools import (
    get_google_search_results,
    get_tavily_formatted_response,
    search_wikipedia_summary,
)

warnings.filterwarnings("ignore")
load_dotenv("local/.env")

TOOLS = [
    {
        "name": "Wikipedia fetcher",
        "description": "Needed to fetch information from Wikipedia, useful for non real time information",
        "func": search_wikipedia_summary,
    },
    {
        "name": "Tavily search",
        "description": "Needed to search information from the web, useful for real time information",
        "func": get_tavily_formatted_response,
    },
    {
        "name": "Google search",
        "description": "Needed to search information from Google, useful for real time information",
        "func": get_google_search_results,
    },
]


class LLM:
    """Class representing the LLM enginge."""

    def __init__(
        self: Self,
        model_name: str,
    ) -> None:
        """Constructor of the LLM model class.

        :param model_name: name of the model
        :type model_name: str
        """

        self.model_name = model_name
        self.llm = self.get_model(model_name=self.model_name)

    st.cache_resource(show_spinner=False)

    @staticmethod
    def get_model(
        model_name: str,
    ) -> ChatOllama:
        """Get the LLM model (from Ollama).

        :param model_name: name of the model
        :type model_name: str
        :return: LLM model
        :rtype: ChatOllama
        """
        return ChatOllama(
            model=model_name,
            temperature=0,
            callbacks=[LLMCallbackHandler()],
        )


class Graph:
    """Class representing the graph with tools and LLM."""

    def __init__(
        self: Self,
        model_name: str,
        tools: list[dict[str, Union[str, callable]]] = TOOLS,
    ) -> None:
        """Constructor of the Graph class.

        :param model_name: name of the model
        :type model_name: str
        :param tools: list of tools
        :type tools: list[dict[str, str | callable]]
        """

        self.model_name = model_name
        self.llm = LLM(model_name=self.model_name).llm
        self.tools = tools
        self.graph_builder = StateGraph(State)

    def get_tools(
        self: Self,
    ) -> list[Tool]:
        """
        Method to assemble tools for the Agent.

        :return: LLM binded with tools
        :rtype: list[Tool]
        """

        return [
            Tool(
                name=tool.get("name"),
                func=tool.get("func"),
                description=tool.get("description"),
            )
            for tool in self.tools
        ]

    def compile_graph(
        self: Self,
    ) -> CompiledStateGraph:
        """
        Method to compile the graph.

        :return: compiled LangGraph graph
        :rtype: CompiledStateGraph
        """

        # Bind LLM with tools
        tools_for_agent = self.get_tools()
        llm_with_tools = self.llm.bind_tools(tools_for_agent)

        self.graph_builder.add_node(
            "chatbot",  # unique name of the node
            partial(
                chatbot, llm=llm_with_tools
            ),  # object called when the node is invoked
        )

        # Add the tool node
        tool_node = BasicToolNode(tools=tools_for_agent)
        self.graph_builder.add_node("tools", tool_node)

        # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
        # it is fine directly responding. This conditional routing defines the main agent loop.
        self.graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            # e.g., "tools": "my_tools"
            {"tools": "tools", END: END},
        )
        # Any time a tool is called, we return to the chatbot to decide the next step
        self.graph_builder.add_edge("tools", "chatbot")
        # The entry point is the chatbot (i.e., chosen LLM)
        self.graph_builder.add_edge(START, "chatbot")

        memory = MemorySaver()

        return self.graph_builder.compile(checkpointer=memory)
