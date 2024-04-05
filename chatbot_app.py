import streamlit as st
import random
from chatbot import ChatBot
import yaml

# Load config from config.yaml
def load_configuration(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_file = 'config.yaml'  
config = load_configuration(config_file)
model_path = config['chatbot_model']['model_path']
rag = config['chatbot_model']['rag']
finetune = config['chatbot_model']['finetune']

# load document list for RAG model
if rag:
    from rag_chatbot import RagChatbot
    document_list = config['rag_config']['documents']

# load new_model_path if using finetuned model
if finetune:
    new_model_path = config['chatbot_model']['finetune_model_path']
else:
    new_model_path = None


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

# Load chatbot
if "chatbot" not in st.session_state:
    if rag:
        st.session_state.chatbot = RagChatbot(model_path, new_model_path, document_list)
    else:
        st.session_state.chatbot = ChatBot(model_path, new_model_path)
    

st.title("Llama 2 Chatbot")

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
    response = st.session_state.chatbot.generate_response(st.session_state.messages)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.session_state.chatbot.output_response(response)
        response = st.write_stream(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
