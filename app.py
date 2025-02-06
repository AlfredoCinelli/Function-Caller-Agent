"""Module building the frontend of the Chatbot assistant via Streamlit."""

import streamlit as st

from src.backend import Graph
from src.utils.logging import logger
from langchain_core.messages import AIMessage
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Agent", layout="wide")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:

    st.session_state.graph = Graph(model_name="qwen2.5").compile_graph()

# Display chat title
st.title("🤖 AI Assistant")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get assistant response
    with st.spinner("🗣️ Calling Agent to get answer..."):
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                config = {"configurable": {"thread_id": "1"}}
                events = st.session_state.graph.stream(
                    {"messages": [("user", prompt)]},
                    config,
                    stream_mode="values"
                )

                # Process streamed response
                for event in events:
                    response = event.get("messages")[-1]
                    if isinstance(response, AIMessage):
                        message_placeholder.markdown(response.content + "▌")

                st.session_state.messages.append({"role": "assistant", "content": response.content})

            except Exception as exc:
                st.error(f"An error occurred: {str(exc)}")


with st.sidebar:
    with st.expander("⚙️ Tools"):
        st.caption("- Wikipedia Search: tool allowing to search for information on Wikipedia.")
        st.caption("- Tavily Web Search: tool allowing to search for information on the web.")
        st.caption("- Google Search: tool allowing to search for information on Google.")
    if st.sidebar.button("Clear Chat", help="Remove the chat history made so far!"):
        st.session_state.messages = []
        st.rerun()