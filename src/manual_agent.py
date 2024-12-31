"""Main script to run the manual ReAct agent."""

# Import packages and modules

import warnings

from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from utils.callbacks import LLMCallbackHandler
from utils.misc import RobustReActOutputParser, find_tool_by_name
from utils.tools import get_text_length

warnings.filterwarnings("ignore")

load_dotenv("local/.env")

if __name__ == "__main__":
    word = input("Enter a word: ")
    print("ReAct Agent")
    tools = [
        get_text_length,
    ]

    # ReAct prompt template (provided by Harisson Chase)
    template = """
    Answer the following questions as best you can. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    # Create prompt and fill in arbitrary values
    prompt = PromptTemplate.from_template(
        template=template,
    ).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # LLM instance
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        stop=["Observation"],
        callbacks=[LLMCallbackHandler()],
    )

    intermediate_steps = []  # type: ignore

    # Agent instance
    agent = (
        {  # type: ignore
            "input": lambda x: x.get("input"),
            "agent_scratchpad": lambda x: format_log_to_str(
                x.get("agent_scratchpad", "")
            ),
        }
        | prompt
        | llm
        | RobustReActOutputParser()
    )

    agent_step = "placeholder"  # type: ignore
    counter, max_iterations = 0, 5
    while not isinstance(agent_step, AgentFinish):
        agent_step: AgentAction | AgentFinish = agent.invoke(  # type: ignore
            {
                "input": f"What is the length in characters of the text {word}?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(tool_input)
            print(f"{tool_name = }")
            print(f"{tool_input = }")
            print(f"{observation = }")
            print(f"{intermediate_steps = }")
            intermediate_steps.append((agent_step, observation))

        counter += 1
        # Stopping condition
        if counter > max_iterations:
            break

    # The final output of the agent should be an istance of AgentFinish class
    if isinstance(agent_step, AgentFinish):
        print(f"Number of ReAct iterations: {counter}")
        print(f"\n{agent_step.return_values.get('output')}")
    else:
        print("\nAgent did not output a final answer!")
