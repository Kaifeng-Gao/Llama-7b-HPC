# Llama-7b-HPC

[Intro to HPC slides](https://docs.google.com/presentation/d/1ZVclDpcvBGjm6CYcPu5WaiwdBvfCX7kjw6cy6tQmZD4/edit#slide=id.g292759f6b3d_0_0) 

[Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

## Preparation

### HPC Environment

1. Move to the compute node by requesting an interactive session for 6 hours: `salloc -t 2:00:00 -G 1 --partition gpu_devel`
2. Load Miniconda module: `module load miniconda`
3. Create an environment for Llama: `conda create --name llama python=3.10`
4. Activate `llama` conda environment: `conda activate llama`
5. Install Pytorch: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
6. Test CUDA using python ![CUDA Test](https://github.com/Kaifeng-Gao/Llama-7b-HPC/blob/main/README.assets/cuda_test.jpg)
7. Install Hugging Face dependencies: 
   1. `conda install conda-forge::transformers` ([tutorial](https://huggingface.co/docs/transformers/installation)）
   2. `conda install -c huggingface -c conda-forge datasets` ([tutorial](https://huggingface.co/docs/datasets/installation))
   3. `conda install -c conda-forge accelerate` ([tutorial](conda install -c conda-forge accelerate))
   4. `pip install bitsandbytes` (Using conda will cause mismatch in CUDA version)
8. Install jupyter notebook for playing around `conda install jupyter`
9. Update OOD
   1. `module unload miniconda`
   2. `ycrc_conda_env.sh update`
10. Exit the compute node to release the job: `exit`

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

## Interactive Chatbot Application 

In addition to the deep learning models and HPC environment setup, this project includes an interactive chatbot application developed with Streamlit. The chatbot, named "K GOD Chatbot," uses the model we have fine-tuned to generate responses to user inputs.

### Prerequisites

Before running the "K GOD Chatbot" application, install Streamlit and other relevant packages using pip:

```bash
pip install streamlit, dotenv
```

### Running the Application

```bash
streamlit run chatbot.py
```

## HPC Related Code

`squeue --me`

`ssh NODELIST`
