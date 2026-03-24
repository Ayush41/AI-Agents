from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import streamlit as st
import os
from dotenv import load_dotenv

# Load API key
# load_dotenv()

# Streamlit page config
st.set_page_config(page_title="LangChain + Gemini Chat", page_icon="🤖")

st.title("🤖 LangChain + Gemini Chat App")
st.write("Ask anything and get responses using Gemini via LangChain")

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    max_tokens=256
)

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