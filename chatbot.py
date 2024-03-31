import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class ChatBot:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_path)
        

    def load_model_and_tokenizer(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        return model, tokenizer

    def generate_prompt_from_history(self, conversation_history):
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

    def generate_response(self, conversation_history):
        history_prompt, length = self.generate_prompt_from_history(conversation_history)
        model_inputs = self.tokenizer(history_prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**model_inputs)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove previous prompts
        response = output[length:]
        return response

    def output_response(self, response):
        for word in response.split():
            yield word + " "
            time.sleep(0.02)
