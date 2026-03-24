from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Simple LangChain program with Gemini integration
# This example demonstrates basic usage of LangChain with Google's Gemini model


# Init the Gemini model
# Make sure you have GOOGLE_API_KEY set in your environment variables
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,  # Controls randomness: 0 = deterministic, 1 = random
    max_tokens=256    # Maximum length of the response
)

# Define a simple prompt
user_query = "What is the capital of France?"

# Create a message object
message = HumanMessage(content=user_query)

# Call the model with the message
response = llm.invoke([message])

# Print the response
print("User Query:", user_query)
print("Model Response:", response.content)