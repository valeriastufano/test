import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import HuggingFaceHub
import os
from getpass import getpass
import pandas as pd

# Set the Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("HF Token:")

# Load data from the CSV file
data = pd.read_csv("supply_chain_data.csv", encoding='ISO-8859-1')

# Create a Pandas Agent with the Hugging Face Hub language model
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
)

# Create a Pandas Agent for analysis
agent = create_pandas_dataframe_agent(llm, data, verbose=True)

# Streamlit app
st.title("Supply Chain Chatbot")

# User input for chatbot
user_input = st.text_input("You:", "")

# Display chat history
chat_history = []

# Process user input and display chatbot response
if st.button("Send"):
    chat_history.append(f"You: {user_input}")
    
    # Execute the user input using the Pandas Agent
    bot_response = agent.run(user_input)
    
    # Display chatbot response
    chat_history.append(f"Bot: {bot_response}")

# Display chat history
st.text("\n".join(chat_history))

# Print statements for debugging
print("HF Token:", os.environ["HUGGINGFACEHUB_API_TOKEN"])
print("Data Shape:", data.shape)
print("Model Configuration:", llm.get_config())
print("User Input:", user_input)
print("Bot Response:", bot_response)
