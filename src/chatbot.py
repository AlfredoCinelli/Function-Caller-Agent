"""Main script to run ChatBot using LangGraph."""

# Import packages and modules

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import Tool
from utils.tools import (
    search_wikipedia_summary,
    get_tavily_formatted_response,
)
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from functools import partial

from dotenv import load_dotenv
from utils.misc import (
    BasicToolNode, # equivalent to ToolNode from langgraph.prebuilt
    State,
    route_tools, # equivalent to tools_condition from langgraph.prebuilt
    chatbot,
)

load_dotenv("local/.env")


graph_builder = StateGraph(State)

llm = ChatOllama(
    model="qwen2.5", # mistral llama3.2:3b qwen2.5 qwen2.5:3b
    temperature=0,  # no randomness
)

tools_for_agent = [
    Tool(
        name="Wikipedia fetcher",  # name of the Agent, it's displayed in the logs
        func=search_wikipedia_summary,  # Python function that will be called by the Agent
        description="Needed to fetch information from Wikipedia, useful for non real time information",  # main component for the LLM to decide which tool to use (necessary!)
    ),
    Tool(
        name="Tavily search",
        func=get_tavily_formatted_response,
        description="Needed to search information from the web, useful for real time information",
    ),
]
llm_with_tools = llm.bind_tools(tools_for_agent)


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", partial(chatbot, llm=llm_with_tools))


tool_node = BasicToolNode(tools=tools_for_agent)
graph_builder.add_node("tools", tool_node)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
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
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        #stream_graph_updates(user_input)
        events = graph.stream(
            {"messages": [("user", user_input)]}, config, stream_mode="values"
        )
        for event in events:
            event["messages"][-1].pretty_print()
    except Exception as e:
        print(f"Something went wrong: {e}")
        break