# Llama-7b-HPC

[Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

# Demo
![alt text](<README.assets/CleanShot 2024-04-04 at 21.43.57@2x.jpg>)

# Quick Start
1. To run the Llama Chatbot, first go through the [Preparation](#preparation), then jump to [Llama 2 Chatbot](#llama2-chatbot)
2. To use 

# Preparation

## Conda Setup Guide

### For Yale HPC Users
Resource for utilizing Yale HPC: [Intro to HPC slides](https://docs.google.com/presentation/d/1ZVclDpcvBGjm6CYcPu5WaiwdBvfCX7kjw6cy6tQmZD4/edit#slide=id.g292759f6b3d_0_0) 

Initiate an interactive session on Yale HPC to install the Conda environment by following these steps:

1. Request an interactive session for 2 hours with GPU access: 
   ```
   salloc -t 2:00:00 -G 1 --partition gpu_devel
   ```
2. Load the Miniconda module: 
   ```
   module load miniconda
   ```

### For Other Users

Ensure Conda is installed on your system. Skip the HPC specific steps above.

## Setting Up Conda Environment

You can create and set up your Conda environment by either using the provided `environment.yml` file or by manually installing each necessary package.

### Using environment.yml (Recommended)

Run the following command to create the environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

### Manual Installation

If the above method does not work for you, follow these steps to manually install your environment and dependencies:

1. **Create and Activate Environment**
   ```
   conda create --name llama python=3.10
   conda activate llama
   ```

2. **Install PyTorch with CUDA Support**
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Test CUDA (Optional)**

   Run the CUDA test Python script
   ```python
   import torch
   torch.version.cuda
   ```
   If it returns a cuda version, the PyTorch with CUDA support is installed successfully.

4. **Install Hugging Face Libraries**

   - Transformers: `conda install conda-forge::transformers`
   - Datasets: `conda install -c huggingface -c conda-forge datasets`
   - Accelerate: `conda install -c conda-forge accelerate`
   - bitsandbytes (for efficient CUDA operations): `pip install bitsandbytes`

5. **Install Jupyter Notebook (Optional)**

   ```
   conda install jupyter
   ```

6. **Install Additional Dependencies**
   - For QLoRA finetuning: `pip install trl peft`
   - For Streamlit applications: `pip install streamlit`
   - For Retrieval Augmented Generation (RAG): `pip install langchain sentence-transformers beautifulsoup4 faiss-gpu`

## Finalizing Setup on Yale HPC

If you are working on Yale HPC, follow these steps to finalize the setup:

1. Unload the Miniconda module:
   ```
   module unload miniconda
   ```

2. Update the OOD (Out-Of-Date) Conda Environment:
   ```
   ycrc_conda_env.sh update
   ```

3. Release the compute node by exiting the session:
   ```
   exit
   ```


## Model Download

1. Apply for access to Llama 2 on [Meta](https://llama.meta.com/llama-downloads) with the same email address used to register Hugging Face
2. Apply for access to Llama 2 model on Hugging Face https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
3. Create a new access token on Hugging Face. Go to your profile -> settings -> API token -> New token
4. Switch to `transfer` node for downloading files: `ssh transfer` (100x download speed than on the compute node)
5. Activate the `llama` environment (Remember to load `miniconda` first): `conda activate llama`
6. Navigate to `palmer_scratch` folder on HPC (Used to store large files)
7. Use the following Python Script to download the model files (Other ways to download model can be found [here](https://huggingface.co/docs/transformers/installation))
   ```python
   from huggingface_hub import snapshot_download
   model_name = "meta-llama/Llama-2-7b-chat-hf"
   access_token = "xxx" #Replace with your own token

   snapshot_download(model_name, cache_dir="./Llama-2-7b-chat-hf", token=access_token)
   ```
8. Snapshot download
   - the snapshot will be under the cache_dir with `models--<organization>--<model-name>/snapshots/<snapshot-id>`
   - for me, the full path is `/home/sds262_<netid>/palmer_scratch/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33`)
9. Set the full snapshot path in [Default Configuration](#default-configuration)

# Configuration Guide

This guide provides an overview of the `config.yaml` configuration file used for configuring a chatbot model based on the Llama 2 7B model with RAG (Retrieval-Augmented Generation) capabilities. The configuration is divided into several sections, each responsible for different aspects of the model's behavior and operation.

## Default Configuration

The `default` configuration provides base settings that other configurations will inherit or override. 

- **model_path**: The path to the main model directory. This path contains the Llama-2-7b-chat-hf model files downloaded in [Preparation](#model-download). 

```yaml
default: &default
  model_path: "/home/sds262_yc958/palmer_scratch/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"
```

## Chatbot Model Configuration

The `chatbot_model` configuration specifies additional settings for the chatbot model that extends or overrides the default configuration.

- **rag**: This setting enables the Retrieval-Augmented Generation feature, which allows the model to use external documents to generate responses.
- **finetune**: Indicates whether fine-tuned model is enabled. If `False`, the model will use its base training without any additional fine-tuning.
- **finetune_model_path**: Specifies the path where the fine-tuned model would be located if fine-tuning was enabled. This setting is not used in this example as fine-tuning is disabled. 
   - Fine-tuned model can be downloaded from #todo
   - Follow the [Fine-tuning](#fine-tune) guide to finetune your own model #todo

```yaml
chatbot_model:
  <<: *default
  rag: True
  finetune: False
  finetune_model_path: "./results/llama-2-7b-sql"
```

## RAG Configuration

The `rag_config` section defines a list of external documents that the chatbot can retrieve information from when generating responses. This is crucial for the RAG functionality to work effectively.

- **documents**: A list of URLs to documents
   - This example is a list of documents related to MySQL reference materials. These documents will be used by the model to pull in relevant information when responding to queries related to MySQL commands and functionalities.

```yaml
rag_config:
  documents: [
    "https://dev.mysql.com/doc/refman/8.0/en/select.html",
    "https://dev.mysql.com/doc/refman/8.0/en/update.html",
    "https://dev.mysql.com/doc/refman/8.0/en/table.html",
    "https://dev.mysql.com/doc/refman/8.0/en/union.html",
    "https://dev.mysql.com/doc/refman/8.0/en/values.html",
    "https://dev.mysql.com/doc/refman/8.0/en/delete.html",
  ]
```


## Inference

1. Clone this repo into `project` folder in HPC
2. Open jupyter from [HPC Dashboard](https://sds262.ycrc.yale.edu/pun/sys/dashboard)
3. Select `llama` (The environment created earlier) from the environment setup dropdown menu
4. Allocate at least 4 CPU each with 8 GiBs RAM
5. Launch the jupyter notebook and open `inference.ipynb` for inference
6. Alternatively, use following Python script or inference_local.py for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# TODO: Replace model path with your own
# The path should look like `models--<organization>--<model-name>/snapshots/<snapshot-id>` under the cache directory defined when downloading the model
model_path = "/home/sds262_kg797/palmer_scratch/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
prompt = "[INST] Tell me about Yale University [/INST]"

model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**model_inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Fine-tune

1. Ensure you have cloned this repo into `project` folder in HPC
2. Install dependencies for QLoRA finetuning, remember to `conda activate llama` before installing: `pip install trl peft`
3. Update OOD
   1. `module unload miniconda`
   2. `ycrc_conda_env.sh update`
4. Select jupyter from [HPC Dashboard](https://sds262.ycrc.yale.edu/pun/sys/dashboard)
5. Select `llama` (The environment created earlier) from the environment setup dropdown menu
6. Allocate at least 1 GPU (Sometimes the RAM may not be sufficient, it depends on the GPU you get)
7. Connect jupyter notebook and open `finetune.ipynb` for finetuning. 
   1. The example in the notebook is for the [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset on Hugging Face.
   2. If using other dataset, need to preprocess the dataset based on [this article](https://huggingface.co/blog/llama2#how-to-prompt-llama-2)

## Llama2 Chatbot

The Llama2 Chatbot is an interactive web application powered by Streamlit, designed to engage users in a conversational interface. It leverages a powerful model to understand and respond to user queries based on a dynamic retrieval of information. A full documentation can be found [here](app_utils/app.md).

### Prerequisites

Before launching the Llama2 Chatbot, it is essential to ensure that your environment is properly set up. Please follow the steps in [Additional Dependency](#setting-up-conda-environment) to install the necessary Dependencies for Streamlit Apps.

### Configuration

To customize the behaviour of the Llama2 Chatbot, certain configurations can be modified in the `config.yaml` file, detailed information can be found in [Configuration Guide](#configuration-guide).
Make sure to adjust these configurations based on your setup and requirements.

### Running the Application
#### For HPC users
If you are an HPC user, here's how to get started:

- Allocate a Remote Desktop with GPU Support: To ensure the Llama2 Chatbot application runs smoothly, use the HPC Open OnDemand service to allocate a remote desktop that supports GPUs ([settings](<README.assets/open_on_demand.jpg>)) (Make sure to select a desktop instance within the `gpu-devel` partition) 
- It's important to note that the model running within the Llama2 Chatbot requires approximately 30GB of GPU memory. Therefore, double-check that the allocated GPU has sufficient capacity by `nvidia-smi`. In some cases, the GPU might not be powerful enough to handle the model efficiently. In this case, delete the connection and allocate again.

#### For other users
- Ensure Your System Has a GPU and a Web Browser: The primary requirement is a GPU with a significant amount of memory, ideally around 30GB, to effectively run the Llama2 Chatbot. 
- Verify that your machine is equipped with such a GPU and that you have access to a web browser for interacting with the application's interface.

With the [prerequisites](#prerequisites) in place and [configurations](#configuration) set, the Llama2 Chatbot application can be launched as follows:

```bash
streamlit run chatbot_app.py
```
