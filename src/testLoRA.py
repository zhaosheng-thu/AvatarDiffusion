from diffusers import StableDiffusionPipeline
import torch
from dm_test import *

model_path = "/root/szhao/diffusers/examples/text_to_image/sd-pokemon-model-lora-szhao-lr-1e-4"
# pipe = StableDiffusionPipeline.from_pretrained("/root/szhao/model-weights/stable-diffusion-v1-4")
# pipe.to("cuda")

prompt = "szhao, handsome person facing the camera" \
                # "studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, " \
                # "dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores"
neg_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting"
image_num = 10
# generate_and_save_img_with_info(_prompt=prompt, _negative_prompt=neg_prompt, num_img=image_num, pipe=pipe, LoRA=False)

pipe_LoRA = StableDiffusionPipeline.from_pretrained("/root/szhao/model-weights/stable-diffusion-v1-4")
print('pipe loaded')
pipe_LoRA.unet.load_attn_procs(model_path)
pipe_LoRA.to("cuda")
print('LoRA Unet loaded')
generate_and_save_img_with_info(_prompt=prompt, 
    # _negative_prompt=neg_prompt, 
    num_img=image_num, pipe=pipe_LoRA, LoRA="LoRA-True")