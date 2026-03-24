from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import streamlit as st

# Simple LangChain program with Gemini integration
# This example demonstrates basic usage of LangChain with Google's Gemini model

# load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")

st.title("🤖 LangChain + Gemini Chat App")
st.write("Ask anything and get responses using Gemini via LangChain")

# Init the Gemini model
# Make sure you have GOOGLE_API_KEY set in your environment variables
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,  # Controls randomness: 0 = deterministic, 1 = random
    max_tokens=256    # Maximum length of the response
)

# # Define a simple prompt
# user_query = "What are the benefits of using LangChain with Gemini for building AI applications?"   
# # with user prompts, you can ask anything you want, and the model will generate a response based on the input.
# user_query = input("Enter your query for the Gemini model: ")

# User input box
user_query = st.text_input("Enter your query:")

# User input box
user_query = st.text_input("Enter your query:")

# Button to trigger response
if st.button("Generate Response"):
    if user_query:
        message = HumanMessage(content=user_query)
        
        with st.spinner("Thinking..."):
            response = llm.invoke([message])
        
        st.subheader("🧑 Your Query:")
        st.write(user_query)

        st.subheader("🤖 Model Response:")
        st.write(response.content)
    else:
        st.warning("Please enter a query!")