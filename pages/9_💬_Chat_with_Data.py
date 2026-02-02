import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os

st.set_page_config(page_title="Chat with Data", page_icon="💬", layout="wide")

st.title("💬 Chat with your Data")
st.markdown("""
Using **Generative AI** (LLMs), you can talk to your dataset in plain English.
Ask questions like:
*   *"What is the average sepal length for each species?"*
*   *"Plot a bar chart of the petal widths."*
*   *"Which flower has the longest sepal?"*
""")

# Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df

df = load_data()

# API Key Handling
api_key = st.text_input("Enter your OpenAI API Key", type="password", help="We do not store this key.")

if not api_key:
    st.warning("Please enter your OpenAI API Key to start chatting.")
    st.info("No API Key? You can get one at https://platform.openai.com/api-keys")
    st.dataframe(df.head())
    st.stop()

# PandasAI Setup
llm = OpenAI(api_token=api_key)
smart_df = SmartDataframe(df, config={"llm": llm})

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = smart_df.chat(prompt)
                
                # PandasAI can return a path to an image (plot) or a string/number
                if response:
                    # Check if response is an image path (basic check)
                    if str(response).endswith(".png") and os.path.exists(str(response)):
                         st.image(response)
                         st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})
                else:
                    st.write("I coudn't verify the answer.")
            except Exception as e:
                st.error(f"Error: {e}")
