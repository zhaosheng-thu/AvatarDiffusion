from diffusers import StableDiffusionPipeline
from PIL import Image
from datetime import datetime
import torch
import os


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_dir_name(num_steps=50, guidance_scale=7.5, scale=512, _prompt="astronaut"):
    base_name_temp = _prompt.replace(", ", " ").split(" ")
    base_name = ""
    for i in range(len(base_name_temp)):
        if 2 < i < 7:
            base_name = base_name + base_name_temp[i] + "_"

    return base_name + "G" + str(guidance_scale) + "_N" + str(num_steps) + "_HW" + str(scale) + "_T" + str(datetime.now().minute) + str(datetime.now().second)


def generate_and_save_img_with_info(num_steps=50, guidance_scale=7.5, scale=512, _prompt="astronaut", num_img=1, _negative_prompt="ugly", pipe=None, LoRA=False):
    # get sub folders name
    sub_folder_name = get_dir_name(num_steps=num_steps, guidance_scale=guidance_scale, scale=scale, _prompt=_prompt)
    os.makedirs(sub_folder_name, exist_ok=True)

    for i in range(num_img):
        print(i)
        num_images = 1
        prompt = [_prompt] * num_images
        negative_prompt = [_negative_prompt] * num_images

        images = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_steps, height=scale,
                      width=scale, negative_prompt=negative_prompt).images

        grid = image_grid(images, rows=1, cols=num_images)

        image_name = str(i + 1) + ".png"
        image_path = os.path.join(sub_folder_name, image_name)
        grid.save(image_path)

    txt_filename = os.path.join(sub_folder_name, "prompts.txt")
    with open(txt_filename, 'w') as file:
        file.write("prompt:" + _prompt + "\nneg_prompt:" + _negative_prompt + "\n" + LoRA + "\n"+ sub_folder_name)


def main():

    # main
    pipe = StableDiffusionPipeline.from_pretrained("/root/szhao/model-weights/stable-diffusion-v1-4")
    pipe = pipe.to("cuda")

    num_steps = 50
    guidance_scale = 9
    img_scale = 512
    num_img = 10
    _prompt_1 = "photo of handsome man standing toward sunlight, wearing glasses " \
                "studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, " \
                "dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores"
    _neg_prompt_1 = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting"

    # _prompt_2 = "A bottle of perfume placed on the table, proportionally arranged, " \
    #             "High definition, rich in detail, well-lit, authentic"
    # _neg_prompt_2 = "Disorganized, low-fidelity, dimly lit"
    list_prompt = [_prompt_1]
    list_neg_prompt = [_neg_prompt_1]
    list_guidance_scale = [7.5]
    list_num_steps = [50]

    for k in range(len(list_num_steps)):
        num_steps = list_num_steps[k]
        for j in range(len(list_guidance_scale)):
            guidance_scale = list_guidance_scale[j]
            for i in range(len(list_prompt)):
                _prompt = list_prompt[i]
                _neg_prompt = list_neg_prompt[i]
                generate_and_save_img_with_info(num_steps=num_steps, guidance_scale=guidance_scale, scale=img_scale,
                                                _prompt=_prompt,
                                                num_img=num_img, _negative_prompt=_neg_prompt, pipe=pipe)

if __name__ == '__main__':
    main()