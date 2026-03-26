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



def main():
    """Run all demonstrations"""
    print("=" * 50)
    print("COMPREHENSIVE LANGCHAIN DEMO WITH GEMINI LLM")
    print("=" * 50)
    
    demo = LangchainDemo()
    
    try:
        #to be implemented and calling these functions in the main function
        demo.basic_llm_call()
        demo.prompt_templates() 
        demo.chat_prompts()
        demo.memory_example()
        demo.sequential_chains()
        # demo.agent_example()  # Uncomment to run
        demo.rag_example()
        demo.output_parsing()
        # demo.streaming_example()  # Uncomment to run
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()