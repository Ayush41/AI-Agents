import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks import StreamingStdOutCallbackHandler

"""
Comprehensive Langchain Program demonstrating core concepts with Gemini LLM
"""


# Import Langchain components

# Load environment variables
load_dotenv()


class LangchainDemo:
    def __init__(self):
        """Initialize Gemini LLM and components"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    # 1. Basic LLM Invocation
    def basic_llm_call(self):
        """Simple LLM invocation"""
        print("\n=== 1. Basic LLM Call ===")
        response = self.llm.invoke("What is Langchain?")
        print(response.content)

    # 2. Prompt Templates
    def prompt_templates(self):
        """Using Prompt Templates"""
        print("\n=== 2. Prompt Templates ===")
        
        # Simple template
        template = PromptTemplate(
            input_variables=["topic"],
            template="Give me 3 interesting facts about {topic}"
        )
        
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.invoke({"topic": "Python"})
        print(result["text"])

    # 3. Chat Prompt Templates
    def chat_prompts(self):
        """Using Chat Prompt Templates"""
        print("\n=== 3. Chat Prompt Templates ===")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful programming assistant."),
            ("human", "Explain {concept} in simple terms")
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.invoke({"concept": "recursion"})
        print(result["text"])

    # 4. Memory Management
    def memory_example(self):
        """Using Conversation Memory"""
        print("\n=== 4. Conversation Memory ===")
        
        memory = ConversationBufferMemory()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=memory)
        
        # Multi-turn conversation
        print(chain.invoke({"input": "Hi, my name is Alice"})["text"])
        print(chain.invoke({"input": "What's my name?"})["text"])

    # 5. Sequential Chains
    def sequential_chains(self):
        """Chaining multiple LLM calls"""
        print("\n=== 5. Sequential Chains ===")
        
        # First chain
        template1 = PromptTemplate(
            input_variables=["topic"],
            template="Generate a creative title about {topic}"
        )
        chain1 = LLMChain(llm=self.llm, prompt=template1, output_key="title")
        
        # Second chain
        template2 = PromptTemplate(
            input_variables=["title"],
            template="Write a short story based on this title: {title}"
        )
        chain2 = LLMChain(llm=self.llm, prompt=template2, output_key="story")
        
        # Sequential chain
        overall_chain = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["topic"],
            output_variables=["title", "story"]
        )
        
        result = overall_chain.invoke({"topic": "AI Revolution"})
        print(f"Title: {result['title']}")
        print(f"Story: {result['story']}")

    # 6. Custom Tools for Agents
    @tool
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    # 7. Agents
    def agent_example(self):
        """Using Agents with tools"""
        print("\n=== 7. Agents with Tools ===")
        
        tools = [
            Tool(
                name="Multiply",
                func=lambda x: int(x.split(",")[0]) * int(x.split(",")[1]),
                description="Multiply two numbers separated by comma"
            ),
            Tool(
                name="Add",
                func=lambda x: int(x.split(",")[0]) + int(x.split(",")[1]),
                description="Add two numbers separated by comma"
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        result = agent.run("What is 5 times 3 plus 10?")
        print(result)

    # 8. Document Loading & RAG
    def rag_example(self):
        """Retrieval Augmented Generation"""
        print("\n=== 8. RAG (Retrieval Augmented Generation) ===")
        
        # Load documents (using example text)
        docs_text = """
        Langchain is a framework for developing applications powered by language models.
        It enables applications to be data-aware and agentic.
        The main value props of LangChain are:
        1. Components: abstractions for working with language models
        2. Chains: assembled components in sequences
        3. Agents: chains that decide which tools to call
        """
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        texts = text_splitter.split_text(docs_text)
        
        # Create vector store
        vectorstore = FAISS.from_texts(texts, self.embeddings)
        
        # Create RAG chain
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        result = qa.run("What are the main value props of LangChain?")
        print(result)

    # 9. Output Parsing
    def output_parsing(self):
        """Parsing structured outputs"""
        print("\n=== 9. Output Parsing ===")
        
        template = PromptTemplate(
            input_variables=["topic"],
            template="List 3 facts about {topic} in format: Fact 1: ..., Fact 2: ..., Fact 3: ..."
        )
        
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.invoke({"topic": "Machine Learning"})
        print(result["text"])

    # 10. Streaming
    def streaming_example(self):
        """Streaming responses"""
        print("\n=== 10. Streaming Responses ===")
        
        streaming_llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            callbacks=[StreamingStdOutCallbackHandler()],
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        streaming_llm.invoke("Write a haiku about programming")


def main():
    """Run all demonstrations"""
    print("=" * 50)
    print("COMPREHENSIVE LANGCHAIN DEMO WITH GEMINI LLM")
    print("=" * 50)
    
    demo = LangchainDemo()
    
    try:
        demo.basic_llm_call() # 1. Basic LLM Call Done✅
        demo.prompt_templates() # 2. Prompt Templates Done✅
        demo.chat_prompts() # 3. Chat Prompt Templates Done✅
        demo.memory_example() # 4. Memory Management Done✅
        demo.sequential_chains() # 5. Sequential Chains Done✅
        # demo.agent_example() 
        demo.rag_example() # 8. RAG (Retrieval Augmented Generation) Done✅
        demo.output_parsing() # 9. Output Parsing Done✅
        # demo.streaming_example()  
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()