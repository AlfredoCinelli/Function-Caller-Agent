"""Main script to run LangChain off-the-shelf agent."""

import warnings

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from utils.callbacks import LLMCallbackHandler

warnings.filterwarnings("ignore")


@tool
def get_text_length(
    text: str,
) -> int:
    """
    Function to returns the length of a text by characters.

    :param text: input text
    :type text: str
    :return: length of the text
    :rtype: int
    """
    to_be_replaced: list[str] = ["\n", "'"]
    for char in to_be_replaced:
        text = text.replace(char, "")

    return len(text)


def text_length_agent(  # type: ignore
    input_word: str,
) -> int:
    """
    Function to returns the length of a text by characters.
    """

    # Define the LLM
    llm = ChatOllama(
        model="mistral",
        temperature=0,  # no randomness
        callbacks=[LLMCallbackHandler()],
    )

    # Define the prompt template
    template = """
    Given the word {input_word}, What is the length of the word?.
    """

    # Create instance of the prompt template
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["input_word"],
    )

    # Define the tools for the agent
    tools_for_agent = [
        Tool(
            name="Text length calculator",  # name of the Agent, it's displayed in the logs
            func=get_text_length,  # Python function that will be called by the Agent
            description="Needed to compute the length of a word",  # main component for the LLM to decide which tool to use (necessary!)
        )
    ]

    # Get the ReAct prompt from LangChain Hub (provided by Harrison Chase)
    react_prompt = hub.pull(
        "hwchase17/react"
    )  # ReAct prompt template (pulled from langchain hub)

    # Create instance of a ReAct agent
    agent = create_react_agent(
        llm=llm,  # initialized LLM runtime
        tools=tools_for_agent,
        prompt=react_prompt,
    )

    # Create an agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=False,
        max_iterations=4,
        handle_parsing_errors=True,
    )

    # Invoke the agent with the filled prompt
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(input_word="Dog")}
    )

    # Get output
    output = result["output"]

    print(f"This is the {output = }")


if __name__ == "__main__":
    word = input("Enter a word: ")
    text_length_agent(word)
