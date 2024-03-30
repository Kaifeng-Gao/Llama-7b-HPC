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

# Function to generate response
def generate_response(model, tokenizer, device, user_input):
    prompt = f"[INST] {user_input} [/INST]"  # Adjust based on how you want to format the prompt
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**model_inputs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Streamed response emulator
def output_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.02)