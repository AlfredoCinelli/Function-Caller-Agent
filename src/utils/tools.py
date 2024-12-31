"""Module aimed at defining the tools to be used by the agents."""

# Import packages and modules
from langchain.agents import tool

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
