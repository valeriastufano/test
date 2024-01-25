#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:00:33 2024

@author: valeriastufano
"""

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
    repo_id="mistralai/Mistral-7B-v0.1", 
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
)

# Create a Pandas Agent for analysis
agent = create_pandas_dataframe_agent(llm, data, verbose=True)

# Execute prompts using the Pandas Agent

agent.run("Write a brief summany of what the data is about")
agent.run("create a pie chart representing the percetage of sales for each product type")