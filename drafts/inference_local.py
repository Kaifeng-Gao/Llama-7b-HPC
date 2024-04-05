from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
from tqdm import tqdm

# TODO: Replace the local model path with your own (downloaded in ./palmer_scratch/)
model_name = "/home/sds262_yl2342/palmer_scratch/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f'Using device: {device}')

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Iterate over each GPU and print its details
    for i in range(num_gpus):
        # Get the device name
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")

        # Get the total memory capacity of the GPU
        total_memory = torch.cuda.get_device_properties(i).total_memory
        print(f"Total memory: {total_memory / 1024**3:.2f} GB")

        # Get the current memory allocated on the GPU
        allocated_memory = torch.cuda.memory_allocated(i)
        print(f"Allocated memory: {allocated_memory / 1024**3:.2f} GB")

        # Get the current memory reserved on the GPU
        reserved_memory = torch.cuda.memory_reserved(i)
        print(f"Reserved memory: {reserved_memory / 1024**3:.2f} GB")

        print("---")
else:
    print("No CUDA-enabled GPU found.")
    

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Interactive Llama-2-7b chatbot. Type 'quit' to exit.")

while True:
    print("================================================================")
    user_input = input("User prompt: ")
    if user_input.lower() == "quit":
        break

    # generate use prompt
    prompt = f"[INST] {user_input} [/INST]"

    # tokenize input
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # time the model inference
    start_time = time.time()

    # Create a progress bar
    progress_bar = tqdm(total=100, unit='step', desc='Generating response')

    def progress_callback(step, total_steps):
        progress_bar.update(1)
        
    # inference and decode 
    output = model.generate(**model_inputs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    end_time = time.time()
    # Close the progress bar
    progress_bar.close()
    
    print("\n")
    print("Llama-7b Assistant:", response)
    inference_time = (end_time - start_time)/60
    print(f"Inference time: {inference_time:.2f} minutes")
    print("\n")