"""Module aimed at defining the callbacks to be used by the LLM for tracing calls and parameters."""

# Import packages and modules
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing_extensions import Self

from src.utils.logging import logger


# Define classes
class LLMCallbackHandler(BaseCallbackHandler):
    """
    Custom class to log the LLM calls and parameters.
    """

    def on_llm_start(  # type: ignore
        self: Self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Log when the LLM starts running.

        :param self: instance variable
        :type self: Self
        :param serialized: the serialized LLM
        :type serialized: dict[str, Any]
        :param prompts: prompt given to the LLM
        :type prompts: list[str]
        :return: print the LLM call and the prompt
        :rtype: None
        """
        logger.info(
            f"LLM starts running.\nIts prompt was:\n{prompts[0]}"
        )  # there only one prompt at this stage

    def on_llm_end(  # type: ignore
        self: Self,
        response: LLMResult,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Log when the LLM ends running.

        :param self: instance variable
        :type self: Self
        :param response: response from the LLM
        :type response: LLMResult
        :return: print the LLM response
        :rtype: None
        """
        logger.info(
            f"LLM ends running.\nIts response was:\n{response.generations[0][0].text}"
        )
