from huggingface_hub import hf_hub_download

# Specify the model repository and filename
repo_id = "CompVis/stable-diffusion-v1-4/vae"
filename = "diffusion_pytorch_model.bin"

# Download the specific file from the Hugging Face Model Hub
local_path = hf_hub_download(repo_id=repo_id, filename=filename)

print(f"File downloaded at {local_path}")