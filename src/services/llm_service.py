"""
LLM service — Gemini chat model client.

Responsibility: initialise and return a configured ChatGoogleGenerativeAI
instance. Nothing else — no prompts, no invocation logic.
"""

from __future__ import annotations

import constant
from settings import config
from utils.Exceptions.errorcodes import client_error
from utils.logger import Logger


class GeminiLlm:
    """Lazily initialises the Gemini chat model."""

    def __init__(self) -> None:
        self.logger = Logger("llm_service")

    # ── Private helpers ────────────────────────────────────────────────────────
    def _get_api_key(self) -> str:
        self.logger.debug("_get_api_key started")
        api_key = config.gemini_api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self.logger.debug("_get_api_key ended")
        return api_key

    # ── Public API ─────────────────────────────────────────────────────────────
    async def fetch_llm(
        self,
        max_tokens: int = constant.MAX_TOKENS,
        temperature: float = constant.TEMPERATURE,
    ):
        """
        Return a configured ``ChatGoogleGenerativeAI`` instance.

        Parameters
        ----------
        max_tokens:  Maximum output token budget for this call.
        temperature: Sampling temperature (0.0 = deterministic).
        """
        self.logger.info(
            "fetch_llm started",
            model=config.gemini_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=config.gemini_model,
                google_api_key=self._get_api_key(),
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            self.logger.info("fetch_llm ended")
            return llm
        except Exception as e:
            await self.logger.error("fetch_llm failed", e)
            raise client_error(str(e))


# Module-level singleton — import this everywhere rather than instantiating.
gemini_llm = GeminiLlm()
