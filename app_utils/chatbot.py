import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
import time


class ChatBot:
    def __init__(self, model_path, new_model_path = None):
        self.device = self.get_device()
        if new_model_path:
            self.model, self.tokenizer = self.load_model_and_tokenizer(new_model_path)
        else:
            self.model, self.tokenizer = self.load_model_and_tokenizer(model_path)
        self.text_generation_pipe = self.init_chain()

    @staticmethod
    def get_device():
        """Returns the device to be used by PyTorch (either CUDA or CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self, model_path):
        """Loads the base transformer model and tokenizer from the given path."""
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        return model, tokenizer
    
    def init_chain(self):
        text_generation_pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.8,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        ) 
        return text_generation_pipeline
    
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
        # model_inputs = self.tokenizer(history_prompt, return_tensors="pt").to(self.device)
        # output = self.model.generate(**model_inputs)
        # output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(history_prompt)
        response = self.text_generation_pipe(history_prompt)

        # Remove previous prompts
        # response = output[length:]
        print(response)
        return response[0]['generated_text']

    def output_response(self, response):
        lines = response.split('\n')
        for line in lines:
            words = line.split(' ')  # Split line into words, keeping spaces in mind
            for word in words[:-1]:  # Yield all but the last word with a trailing space
                yield word + " "  
                time.sleep(0.02)
            if words:  # Check if the line was not empty
                yield words[-1]  # Yield the last word without adding an extra space
                time.sleep(0.02)
            yield '\n'  # At the end of a line, yield a newline character