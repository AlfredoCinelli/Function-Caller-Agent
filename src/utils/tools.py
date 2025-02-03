"""Module aimed at defining the tools to be used by the agents."""

# Import packages and modules
from tavily import TavilyClient
from langchain.agents import tool
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv("local/.env")

# Tools definition


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
    to_be_replaced: list[str] = ["\n", "'", '"']
    for char in to_be_replaced:
        text = text.replace(char, "")

    return len(text)

@tool
def search_wikipedia_summary(
    query: str,
) -> str:
    """Search Wikipedia for the given query and return a summary."""
    wikipedia_res = WikipediaLoader(
        query=query,
        load_max_docs=3,
        doc_content_chars_max=2_000,
    ).load()
    
    wikipedia_summary = "\n\n".join([
        doc.metadata.get("summary", "") for doc in wikipedia_res
    ])
    
    #wikipedia_summary = wikipedia_res[0].metadata.get("summary", "")
    
    return wikipedia_summary

@tool
def get_tavily_search_results(
    query: str,
) -> str:
    """Search Tavily for the given query and return a summary."""
    tavily_search_results = TavilySearchResults(
        max_results=1,
    )
    return tavily_search_results.invoke(query)

@tool
def get_tavily_formatted_response(
    query: str,
) -> str:
    """Use Tavily to search for the given query and return a formatted response."""
    
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    return tavily_client.qna_search(query=query)
    
