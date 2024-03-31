from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
# import os
# from dotenv import load_dotenv

# # Retrieve the access token
# load_dotenv()
# access_token = os.getenv('ACCESS_TOKEN')

# Function to load model and tokenizer
def load_model_and_tokenizer():
    model_path = "/home/sds262_kg797/palmer_scratch/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    return model, tokenizer, device

# Function to generate current prompt from conversation history
def generate_prompt_from_history(conversation_history):
    history_prompt = ""
    length = 0
    for message in conversation_history:
        if message["role"] == "user":
            history_prompt += f"<s>[INST] {message['content']} [/INST]"
            length += len(message['content']) + 16
        elif message["role"] == "assistant":
            history_prompt += f" {message['content']} </s>"
            length += len(message['content']) + 2

    return history_prompt, length

# Function to generate response
def generate_response(model, tokenizer, device, history_prompt, length):
    model_inputs = tokenizer(history_prompt, return_tensors="pt").to(device)
    output = model.generate(**model_inputs)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove previous prompts
    response = output[length:]

    return response

# Streamed response emulator
def output_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.02)