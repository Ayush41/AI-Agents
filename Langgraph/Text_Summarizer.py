import re

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
                "You are a text summarization agent. "
                "Read the input text and return a short, informative summary. "
                "Keep the summary concise and easy to understand.\n\n{text}"
            )
            prompt = Prompt(text=prompt_text, input_variables=["text"])
            llm = OpenAI(model_name=model_name)
            self.agent = Agent(prompt=prompt, llm=llm)

    def summarize(self, text: str) -> str:
        if self.agent:
            try:
                return self.agent.run({"text": text})
            except Exception:
                return self._fallback_summary(text)
        return self._fallback_summary(text)

    def _fallback_summary(self, text: str) -> str:
        if not text.strip():
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= 2:
            return text.strip().replace("\n", " ")

        summary = " ".join(sentences[:2]).strip()
        if len(summary) > 280:
            summary = summary[:280].rsplit(" ", 1)[0].strip() + "..."
        return summary


def collect_input() -> str:
    print("Enter the text you want to summarize. Press Enter twice to finish.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    source_text = collect_input()
    if not source_text:
        print("No text provided. Exiting.")
    else:
        agent = TextSummarizerAgent()
        summary = agent.summarize(source_text)
        print("\nSummary:\n")
        print(summary)
