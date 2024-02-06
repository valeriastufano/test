import os
import json
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import HuggingFaceHub
import pandas as pd

# Set the Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def create_agent(data):
    print("Attempting to create HuggingFaceHub instance...")
    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
        )
        print("HuggingFaceHub instance created successfully.")
    except Exception as e:
        print(f"Error creating HuggingFaceHub instance: {e}")
        raise e

    # Create a Pandas Agent for analysis
    agent = create_pandas_dataframe_agent(llm, data, verbose=True)
    return agent

def ask_agent(agent, query):
    prompt = """
        Let's decode the way to respond to the queries...
        Your query: {}
    """.format(query)

    response = agent.run(prompt)
    return str(response)

def decode_response(response):
    return json.loads(response)

def write_answer(response_dict):
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Handle "bar" response type
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            st.bar_chart(data)
        except ValueError:
            st.error("Error displaying bar chart")

    # Handle "pie_chart" response type
    if "pie_chart" in response_dict:
        data = response_dict["pie_chart"]
        try:
            st.pie_chart(data)
        except ValueError:
            st.error("Error displaying pie chart")

    # Handle "table" response type
    if "table" in response_dict:
        data = response_dict["table"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)
        except ValueError:
            st.error("Error displaying table")

    # Handle "graph" response type (you may need to customize this based on your use case)
    if "graph" in response_dict:
        data = response_dict["graph"]
        try:
            # Add code to display the graph based on your specific format
            st.write("Displaying graph:", data)
        except ValueError:
            st.error("Error displaying graph")

# Streamlit app
st.set_page_config(page_title="Supply Chain Data Analyzer")
st.title("Supply Chain Data Analyzer")

# File upload
data = st.file_uploader("Upload a CSV file", type="csv")

# Check if a file is uploaded
if data is not None:
    # Load data from the uploaded file
    data_df = pd.read_csv(data, encoding='ISO-8859-1')

    # Create an agent from the uploaded CSV file.
    agent = create_agent(data_df)

    # Query input
    query = st.text_area("Ask a question about the data")

    if st.button("Submit Query", type="primary"):
        # Query the agent.
        response = ask_agent(agent=agent, query=query)

        # Decode the response.
        decoded_response = decode_response(response)

        # Write the response to the Streamlit app.
        write_answer(decoded_response)

        st.success("Query executed successfully!")
else:
    st.warning("Please upload a CSV file.")
