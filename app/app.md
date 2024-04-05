# Llama 2 Chatbot Application

> Below is a detailed documentation for the front-end display and the principles behind the chatbot application as described by the three Python files (chatbot.py, rag_chatbot.py, and chatbot_app.py).

## Overview
The Llama 2 Chatbot is an interactive web application powered by Streamlit. It utilizes cutting-edge language models to engage users in conversation. The application offers two modes:

1. A standard chat mode using a causal language model for generating responses.
2. An advanced mode that employs a Retriever-Augmented Generation (RAG) approach, enhancing responses with information retrieved from a set of documents.

## Front-End Display
The user interface is clean and simple, built with Streamlit. It features:

- A title at the top of the page.
- A dynamically chosen greeting message that prompts user input.
- An area where the conversation history is displayed, showing messages from both the user and the chatbot.
- A text input field where the user can type their messages.


## Principles of Operation

### ChatBot Class (chatbot.py)
This module defines the ChatBot class, responsible for:

- Initializing the model and tokenizer based on a given model_path.
- Generating prompts from the conversation history, ensuring they are correctly formatted.
- Producing responses from the model using the generated prompts.
- Streaming the response to the interface, simulating real-time typing for an enhanced user experience.

### RagChatbot Class (rag_chatbot.py)
This module extends the ChatBot class to introduce document retrieval capabilities:

- Upon initialization, it loads a set of documents to use as reference material for generating responses.
- It utilizes the HuggingFacePipeline and a chain of language model processing (llm_chain) to generate responses informed by the context retrieved from the documents.

### Chatbot Application (chatbot_app.py)
This is the main file that orchestrates the web application:

- It loads the configuration from a YAML file, which specifies the model to use and the documents list if in RAG mode.
- Initializes the appropriate chatbot class (either ChatBot or RagChatbot) based on the configuration.
- Manages the session state to maintain the conversation history and handle user interactions.
- Displays the conversation and manages the input and output flow of the chatbot interaction.

## Interaction Flow

1. When a user visits the application, they are greeted with a random greeting message.
2. The user enters their message into the input field.
3. The app adds the user's message to the conversation history and passes it to the chatbot.
4. The chatbot generates a response, which is then displayed to the user.
5. The response is added to the conversation history and the cycle continues with each new message.

## Configurable Options
The application can be configured to work in two modes:

- Standard Chat Mode: Uses the language model directly for chat interactions.
- RAG Mode: Enhances the chatbot responses with contextual information from the provided documents.

## Conclusion
The Llama 2 Chatbot represents an advanced application of natural language processing, providing a user-friendly interface for complex interaction with AI models. Whether it's casual conversation or information-dense dialogue, the chatbot adapts to user inputs with contextually aware responses.