import streamlit as st
import random

from model_utils import load_model_and_tokenizer, generate_prompt_from_history, generate_response, output_response

# Randomly choose a greeting if not already chosen
if "greeting" not in st.session_state:
    st.session_state.greeting = random.choice([
        "What's up?",
        "How's it going?",
        "What's on your mind?",
        "How are you today?",
        "Anything new with you?",
        "How's your day going?",
        "What can I do for you today?",
        "How can I assist you right now?",
        "Ask me anything!",
        "Anything I can help with?",
        "What are you up to?",
        "How's everything?"
    ])

# Load model and tokenizer once
if "model" not in st.session_state:
    st.session_state.model, st.session_state.tokenizer, st.session_state.device = load_model_and_tokenizer()

st.title("K GOD Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(st.session_state.greeting):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the prompt from the conversation history
    history_prompt = generate_prompt_from_history(st.session_state.messages)
    response = generate_response(st.session_state.model, st.session_state.tokenizer, st.session_state.device, history_prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(output_response(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
