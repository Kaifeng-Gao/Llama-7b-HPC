import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import yaml

def load_configuration(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_file = 'config.yaml'  
config = load_configuration(config_file)
model_path = config['chatbot_model']['model_path']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

parser = argparse.ArgumentParser(description='Generate text from a prompt')
parser.add_argument('prompt', type=str, help='The prompt to generate text from')

args = parser.parse_args()
prompt = f"[INST] {args.prompt} [/INST]"

model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**model_inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))