# Llama-7b-HPC

[Intro to HPC slides](https://docs.google.com/presentation/d/1ZVclDpcvBGjm6CYcPu5WaiwdBvfCX7kjw6cy6tQmZD4/edit#slide=id.g292759f6b3d_0_0) 

[Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

## Preparation

### Conda Setup Guide

#### For Yale HPC Users

Initiate an interactive session on Yale HPC to install the Conda environment by following these steps:

1. Request an interactive session for 2 hours with GPU access: 
   ```
   salloc -t 2:00:00 -G 1 --partition gpu_devel
   ```
2. Load the Miniconda module: 
   ```
   module load miniconda
   ```

#### For Other Users

Ensure Conda is installed on your system. Skip the HPC specific steps above.

### Setting Up Conda Environment

You can create and set up your Conda environment by either using the provided `environment.yml` file or by manually installing each necessary package.

#### Using environment.yml (Recommended)

Run the following command to create the environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

#### Manual Installation

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

   Run the CUDA test Python script found [here](https://github.com/Kaifeng-Gao/Llama-7b-HPC/blob/main/README.assets/cuda_test.jpg).

4. **Install Hugging Face Libraries**

   - Transformers: `conda install conda-forge::transformers`
   - Datasets: `conda install -c huggingface -c conda-forge datasets`
   - Accelerate: `conda install -c conda-forge accelerate`
   - bitsandbytes (for efficient CUDA operations): `pip install bitsandbytes`

5. **Install Jupyter Notebook**

   ```
   conda install jupyter
   ```

6. **Install Additional Dependencies (Optional)**
   - For QLoRA finetuning: `pip install trl peft`
   - For Streamlit applications: `pip install streamlit python-dotenv`
   - For Retrieval Augmented Implementation: `pip install langchain sentence-transformers beautifulsoup4 faiss-gpu`

### Finalizing Setup on Yale HPC

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

### Model Download

1. Apply for access to Llama 2 on [Meta](https://llama.meta.com/llama-downloads) with the same email address used to register Hugging Face
2. Apply for access to Llama 2 model on Hugging Face https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
3. Create a new access token on Hugging Face. Go to your profile -> settings -> API token -> New token
4. Switch to `transfer` node for downloading files: `ssh transfer` (100x download speed than on the compute node)
5. Activate the `llama` environment (Remember to load `miniconda` first): `conda activate llama`
6. Navigate to `palmer_scratch` folder on HPC (Used to store large files)
7. Use the following Python Script to download the model files (Other ways to download model can be found [here](https://huggingface.co/docs/transformers/installation))
8. Snapshot download (the snapshot will be under the cache_dir with `models--<organization>--<model-name>/snapshots/<snapshot-id>`, for me, the full path is `/home/sds262_<netid>/palmer_scratch/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33`)
```python
from huggingface_hub import snapshot_download
model_name = "meta-llama/Llama-2-7b-chat-hf"
access_token = "xxx" #Replace with your own token

snapshot_download(model_name, cache_dir="./Llama-2-7b-chat-hf", token=access_token)
```

![Model downloading](https://github.com/Kaifeng-Gao/Llama-7b-HPC/blob/main/README.assets/model_download.jpg)

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

The Llama2 Chatbot is an interactive web application powered by Streamlit, designed to engage users in a conversational interface. It leverages a powerful model to understand and respond to user queries based on a dynamic retrieval of information.

### Prerequisites

Before launching the Llama2 Chatbot, it is essential to ensure that your environment is properly set up. Please follow the steps in Additional Dependencies (Optional) to install the necessary Dependencies for Streamlit Apps.

### Configuration

To customize the behaviour of the Llama2 Chatbot, certain configurations can be modified in the `config.yaml` file. These include:

- `model_path`: Specifies the path to the model weights.
- `rag`: A boolean indicator (`True` or `False`) to decide whether to use the Retrieval Augmented Generation model. This setting enhances the chatbot's response quality by integrating retrieved documents into its generation process.
- `documents`: Defines a list of websites from which the model can dynamically acquire information to inform its responses. This will only work when `rag` is set to `True`.

Make sure to adjust these configurations based on your setup and requirements.

### Running the Application

With the prerequisites in place and configurations set, the Llama2 Chatbot application can be launched as follows:

```bash
streamlit run chatbot_app.py
```
