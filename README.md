# Llama-7b-HPC

# Demo
![Llama 2 Chatbot](<demo/Llama 2 Chatbot.jpg>)
---
🔍 Retrieval Augmented: Wave goodbye to endless searches! Our system delivers quick access to HTML document lists, making information retrieval a breeze.

🌐 Interactive Web App: Dive into an immersive experience with our web application powered by Streamlit. Enjoy chatting with beautiful Markdown formatting.

🤖 Fine-tuned Model: Elevate your SQL programming with our fine-tuned model, versatile and open to embracing other models for an unmatched experience.

📘 Yale HPC Guide: Your gateway to mastery! Full documentation at your fingertips for reproduction on Yale High-Performance Computing.

# TOC
- [Preparation](#preparation)
- [Configuration Guide](#configuration-guide)
- [Llama2 Chatbot](#llama2-chatbot)
- [Inference](#inference)
- [Fine-tune](#fine-tune)
- [Reference](#reference)

# Quick Start
1. To run the Llama Chatbot, first go through the [Preparation](#preparation), then jump to [Llama 2 Chatbot](#llama2-chatbot).
2. To fine-tune your own model, first go through the [Preparation](#preparation), then follow the guide in [Fine-tune](#fine-tune).
3. For command line quick inference, go through [Preparation](#preparation) and then jump to [Inference](#inference).

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


## Base Model Download

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

## Fine-tuned Model Download

#todo

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

## Finetuning Configuration
The `model_config` section contains the model used in fine-tuning
 - **<<: *default**: This line inherits configurations from [Default](#default-configuration)
  - **access_token**: The `access_token` is essential for accessing datasets used for fine-tuning on Hugging Face's platform.
  - **new_model**: This path `"./models/llama-2-7b-sql"` indicates where the fine-tuned model will be saved locally after the fine-tuning process is completed.
  - **dataset_name**: Indicates the name of the dataset that will be used for fine-tuning. Here, we use [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) from hugging face as an example
```yaml
model_config: 
  <<: *default
  access_token: <your token>
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  new_model: "./results/llama-2-7b-sql"
  dataset_name: "b-mc2/sql-create-context"
```

Detailed explanation for `q_lora_parameters`, `bitsandbytes_parameters`, `training_arguments` and `sft_parameters` can be found [here](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing).


# Llama2 Chatbot

The Llama2 Chatbot is an interactive web application powered by Streamlit, designed to engage users in a conversational interface. It leverages a powerful model to understand and respond to user queries based on a dynamic retrieval of information. A full documentation can be found [here](app_utils/app.md).

## Prerequisites

Before launching the Llama2 Chatbot, it is essential to ensure that your environment is properly set up. Please follow the steps in [Additional Dependency](#setting-up-conda-environment) to install the necessary Dependencies for Streamlit Apps.

## Configuration

To customize the behaviour of the Llama2 Chatbot, certain configurations can be modified in the `config.yaml` file, detailed information can be found in [Configuration Guide](#chatbot-model-configuration).
Make sure to adjust these configurations based on your setup and requirements.

## Running the Application
### For HPC users
If you are an HPC user, here's how to get started:

- Allocate a Remote Desktop with GPU Support: To ensure the Llama2 Chatbot application runs smoothly, use the HPC Open OnDemand service to allocate a remote desktop that supports GPUs ([settings](<README.assets/open_on_demand.jpg>)) (Make sure to select a desktop instance within the `gpu-devel` partition) 
- It's *IMPORTANT* to note that the model running within the Llama2 Chatbot requires approximately 30GB of GPU memory. Therefore, double-check that the allocated GPU has sufficient capacity by `nvidia-smi`. In some cases, the GPU might not be powerful enough to handle the model efficiently. In this case, delete the connection and allocate again.
   - An alternative approach is to use xQuartz as graphical interface instead ([tutorial](https://docs.ycrc.yale.edu/clusters-at-yale/access/x11/)), this way, the gpu type can be designated in job allocation like this `--gpus a100:1`

### For other users
- Ensure Your System Has a GPU and a Web Browser: The primary requirement is a GPU with a significant amount of memory, ideally around 30GB, to effectively run the Llama2 Chatbot. 
- Verify that your machine is equipped with such a GPU and that you have access to a web browser for interacting with the application's interface.

With the [prerequisites](#prerequisites) in place and [configurations](#configuration) set, the Llama2 Chatbot application can be launched as follows:

```bash
streamlit run chatbot_app.py
```

# Inference
For Web-based inference, turn to [Chatbot](#llama2-chatbot).

For command line quick inference:

1. Modify the `model_path` in `config.yaml` based on [Configuration Guide](#default-configuration)
2. Activate the conda environment
3. Run the script `python inference.py "your prompt"`

```bash
(Llama)[sds262_kg797@r805u30n01.grace Llama-7b-HPC]$ python inference.py "Tell me about Yale"
Using device: cuda
Loading checkpoint shards: 100%
[INST] Tell me about Yale [/INST]  Yale University is a private Ivy League research university located in New Haven, Connecticut, United States. It was founded in 1701 and is one of the oldest institutions of higher learning in the United States. Yale is known for its academic excellence, cultural influence, and historic prestige.

Here are some key facts about Yale:

1. History: Yale was founded in 1701 as the Collegiate School, and it was renamed Yale College in 1718 in recognition of a gift from Elihu Yale, a British merchant and early benefactor of the institution.
...
```

# Fine-tune
1. Modify configurations based on [Configuration Guide](#finetuning-configuration)
2. If using other dataset: also need to update `finetune_utils.dataset_converter` to construct instructions for fine-tuning. The template for prompting Llama can be found [here](https://huggingface.co/blog/llama2#how-to-prompt-llama-2)
3. For HPC users, use batch job to run the fine-tuning script
   ```bash
   sbatch run_finetune.sh
   ```
4. For other users, run the python script for fine-tuning
   ```bash
   python finetune.py
   ```


# Reference
1. [Llama2](https://huggingface.co/docs/transformers/model_doc/llama2) 
2. [Build your own RAG with Mistral-7B and LangChain | by Madhav Thaker | Medium](https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146)
3. [Build a basic LLM chat app - Streamlit Docs](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
4. [Fine-tune the “Llama-v2-7b-guanaco” model with 4-bit QLoRA and generate Q&A datasets from PDFs](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing)