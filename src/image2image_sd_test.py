from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from datetime import datetime
import torchvision.transforms as transforms
import os
from text2image_sd_test import image_grid, get_dir_name


def image_loader(image_name):
    image = Image.open(image_name)
    image = transforms.ToTensor()(image).unsqueeze(0) # 给张量增添一个第零维度
    return image


def main():
    
    image2image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/root/szhao/model-weights/stable-diffusion-v1-4")
    image2image_pipe = image2image_pipe.to("cuda")
    print("Model loaded")
    
    num_steps = 35
    guidance_scale = 7.5
    strength = 0.2
    num_img = 10
    load_path = '/root/szhao/Diffusion/figs/boy_standing_toward_sunlight_G7.5_N40_HW512_T1811/1.png'
    image_input = image_loader(load_path)
    
    prompt = "a handsome boy, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores"
    neg_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting"
    
    subfolder_name = get_dir_name(num_steps=num_steps, guidance_scale=guidance_scale, scale=512, _prompt=prompt)
    os.makedirs(subfolder_name, exist_ok=True)
    for num in range(num_img):
        images = image2image_pipe(prompt=prompt, strength=strength, image=image_input, guidance_scale=guidance_scale,
                                num_inference_steps=num_steps, negative_prompt=neg_prompt).images

        grid = image_grid(images, rows=1, cols=1)
        print(num)
        image_name = str(num + 1) + ".png"
        image_path = os.path.join(subfolder_name, image_name)
        grid.save(image_path)


if __name__ == "__main__":
    main()