import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from audio_recorder_streamlit import audio_recorder
import os
from openai import OpenAI as OpenAIClient # Official client for Whisper

st.set_page_config(page_title="Chat with Data", page_icon="💬", layout="wide")

st.title("💬 Chat with your Data (Voice Enabled 🎙️)")
st.markdown("""
Ask questions in plain English or **use your voice**.
*   *"Show me the distribution of species"*
*   *"Which flower has the widest sepal?"*
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
api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Required for both Chat and Voice.")

if not api_key:
    st.warning("Please enter your OpenAI API Key to start.")
    st.stop()

# ---------------------------------------------------------
# Voice Input
# ---------------------------------------------------------
audio_bytes = audio_recorder(text="Click to Record Request", icon_size="2x")
transcribed_text = None

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    
    # Transcribe via Whisper
    with st.spinner("Transcribing audio..."):
        try:
            client = OpenAIClient(api_key=api_key)
            with open("temp_audio.wav", "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            transcribed_text = transcript.text
            st.success(f"Transcribed: '{transcribed_text}'")
        except Exception as e:
            st.error(f"Whisper Error: {e}")

# ---------------------------------------------------------
# Chat Logic
# ---------------------------------------------------------
llm = OpenAI(api_token=api_key)
smart_df = SmartDataframe(df, config={"llm": llm})

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Determine Input Source (Voice or Text)
prompt = None
# Prioritize Voice if just happened, otherwise Check Chat Input
if transcribed_text:
    prompt = transcribed_text
    # Reset audio bytes roughly (streamlit helps reset on rerun but we handle flow here)

# Standard Text Input (always visible)
text_input = st.chat_input("...or type your question here")
if text_input:
    prompt = text_input

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = smart_df.chat(prompt)
                
                if response:
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
