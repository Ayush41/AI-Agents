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


def render_ui() -> None:
    st.set_page_config(page_title="Text Summarizer AI", page_icon="📝", layout="wide")

    # st.markdown(
    #     """
    #     # <style>
    #     # .stApp { background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%); }
    #     # .block-container { padding-top: 2rem; }
    #     # .card {
    #     #     background: white;
    #     #     border-radius: 16px;
    #     #     padding: 1.2rem;
    #     #     box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    #     #     border: 1px solid #e2e8f0;
    #     # }
    #     # </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.title("📝 Smart Text Summarizer")
    st.caption("Paste long content and get a crisp, readable summary in seconds.")

    with st.sidebar:
        st.header("Options")
        style = st.radio("Summary style", ["Concise", "Balanced", "Detailed"], horizontal=False)
        st.markdown("---")
        st.subheader("Tips")
        st.write("- Paste articles, notes, or meeting transcripts.")
        st.write("- Use a concise style for quick overviews.")
        st.write("- Use detailed mode when you want more context.")
        if st.button("Load example", use_container_width=True):
            st.session_state["source_text"] = (
                "Artificial intelligence is reshaping how people work, learn, and communicate. "
                "Modern tools can now summarize documents, answer questions, and automate repetitive tasks. "
                "This makes everyday work faster while creating new opportunities for creativity and decision-making."
            )

    col1, col2 = st.columns([1.4, 0.8], gap="large")
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        source_text = st.text_area(
            "Enter your text",
            key="source_text",
            height=280,
            placeholder="Paste the text you want summarized here...",
            value=st.session_state.get("source_text", ""),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("What happens next")
        st.write("The assistant reads your text and creates a clean summary that keeps the important points.")
        st.write("If the AI backend is unavailable, the app gracefully falls back to a built-in summarizer.")
        st.markdown("---")
        if st.button("✨ Summarize now", use_container_width=True, type="primary"):
            if source_text.strip():
                with st.spinner("Preparing your summary..."):
                    agent = TextSummarizerAgent()
                    summary = agent.summarize(source_text, style=style.lower())
                st.session_state["summary"] = summary
            else:
                st.session_state["summary"] = ""
                st.warning("Please enter some text before summarizing.")
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["source_text"] = ""
            st.session_state["summary"] = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("summary"):
        st.markdown("### Summary")
        st.info(st.session_state["summary"])
    elif st.session_state.get("summary") == "":
        st.caption("Your summary will appear here once you submit text.")


if __name__ == "__main__":
    render_ui()
