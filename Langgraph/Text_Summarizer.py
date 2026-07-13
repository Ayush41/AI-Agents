import re
from typing import Optional

import streamlit as st

try:
    from langgraph import Agent
    from langgraph.llms import OpenAI
    from langgraph.prompts import Prompt

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class TextSummarizerAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.agent = None
        if LANGGRAPH_AVAILABLE:
            prompt_text = (
                "You are a polished text summarization assistant. "
                "Read the input text and return a short, informative summary. "
                "Keep the summary clear, readable, and easy to understand.\n\n{text}"
            )
            prompt = Prompt(text=prompt_text, input_variables=["text"])
            llm = OpenAI(model_name=model_name)
            self.agent = Agent(prompt=prompt, llm=llm)

    def summarize(self, text: str, style: str = "balanced") -> str:
        if self.agent:
            try:
                return self.agent.run({"text": text})
            except Exception:
                return self._fallback_summary(text, style)
        return self._fallback_summary(text, style)

    def _fallback_summary(self, text: str, style: str = "balanced") -> str:
        if not text.strip():
            return ""

        cleaned_text = re.sub(r"\s+", " ", text.strip())
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        target_count = {"concise": 2, "balanced": 3, "detailed": 4}.get(style.lower(), 3)
        selected = sentences[: min(target_count, len(sentences))]
        summary = " ".join(selected)

        if len(summary) > 320:
            summary = summary[:320].rsplit(" ", 1)[0].strip() + "..."
        return summary

