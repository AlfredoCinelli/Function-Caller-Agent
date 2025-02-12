"""Module aimed at defining the tools to be used by the agents."""

# Import packages and modules
import os

from dotenv import load_dotenv
from langchain.agents import tool
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from tavily import TavilyClient

from src.utils.logging import logger

load_dotenv("local/.env")

# Tools definition


@tool(parse_docstring=False)
def search_wikipedia_summary(
    query: str,
) -> str:
    """
    Search Wikipedia for the given query and return a summary.

    :param query: query to search for
    :type query: str
    :return: summary of retrieved Wikipedia pages
    :rtype: str
    """
    logger.info("Calling Wikipedia fetcher tool.")
    wikipedia_res = WikipediaLoader(
        query=query,
        load_max_docs=3,
        doc_content_chars_max=2_000,
    ).load()

    wikipedia_summary = "\n\n".join(
        [doc.metadata.get("summary", "") for doc in wikipedia_res]
    )

    return wikipedia_summary


@tool(parse_docstring=False)
def get_tavily_formatted_response(
    query: str,
) -> str:
    """
    Search Tavily for the given query and return a summary.

    :param query: query to search for
    :type query: str
    :return: retrieved Tavily search results
    :rtype: str
    """
    logger.info("Calling Tavily search tool.")
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    return tavily_client.qna_search(query=query)


@tool(parse_docstring=False)
def get_google_search_results(
    query: str,
) -> str:
    """
    Search Google for the given query and return a response.

    :param query: query to search for
    :type query: str
    :return: retrieved Google search results
    :rtype: str
    """
    logger.info("Calling Google search tool.")
    google_search = GoogleSerperAPIWrapper()

    return google_search.run(query)
