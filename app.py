import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import Tool
from src.utils.tools import search_wikipedia_summary, get_tavily_formatted_response
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from functools import partial
from src.utils.misc import BasicToolNode, State, route_tools, chatbot
import time

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'graph' not in st.session_state:
        st.session_state.graph = create_graph()

def create_graph():
    """Create and return the LangGraph chatbot graph."""
    graph_builder = StateGraph(State)
    
    # Initialize LLM
    llm = ChatOllama(
        model="mistral-nemo", # qwen2.5
        temperature=0,
    )
    
    # Define tools
    tools_for_agent = [
        Tool(
            name="Wikipedia fetcher",
            func=search_wikipedia_summary,
            description="Needed to fetch information from Wikipedia, useful for non real time information",
        ),
        Tool(
            name="Tavily search",
            func=get_tavily_formatted_response,
            description="Needed to search information from the web, useful for real time information",
        ),
    ]
    
    llm_with_tools = llm.bind_tools(tools_for_agent)
    
    # Add nodes
    graph_builder.add_node("chatbot", partial(chatbot, llm=llm_with_tools))
    tool_node = BasicToolNode(tools=tools_for_agent)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    # Compile graph
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def display_messages():
    """Display chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def process_user_message(user_input):
    """Process user input and get bot response."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response
    config = {"configurable": {"thread_id": "1"}}
    events = st.session_state.graph.stream(
        {"messages": [("user", user_input)]}, 
        config, 
        stream_mode="values"
    )
    
    # Process and display bot response
    for event in events:
        response = event["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display thinking animation
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for i in range(len(response)):
                message_placeholder.markdown(response[:i+1] + "▌")
                time.sleep(0.01)
            message_placeholder.markdown(response)

def main():
    # Page config
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    # Header
    st.header("🤖 AI Assistant")
    st.markdown("""
    This AI assistant can help you with various tasks including:
    - Searching Wikipedia
    - Finding real-time information
    - Answering questions and having conversations
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    display_messages()
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        try:
            process_user_message(user_input)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.button("Reset Chat"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()