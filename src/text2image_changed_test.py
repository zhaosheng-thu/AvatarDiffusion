from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from tqdm.auto import tqdm
import torch
import os
from torchvision import transforms
from PIL import Image

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# vae = AutoencoderKL.from_pretrained("/root/szhao/model-weights/stable-diffusion-v1-4", subfolder="vae")
print('vae loaded')

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
# tokenizer = CLIPTokenizer.from_pretrained("E:\DLmodels\DM\clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("/root/szhao/model-weights/clip-vit-large-patch14")
# text_encoder = CLIPTextModel.from_pretrained("E:\DLmodels\DM\clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("/root/szhao/model-weights/clip-vit-large-patch14")
print('tokenize and text_encoder loaded')

# 3. The UNet model for generating the latents.
# unet = UNet2DConditionModel.from_pretrained("E:\DLmodels\DM\stable-diffusion-v1-4", subfolder="unet")
unet = UNet2DConditionModel.from_pretrained("/root/szhao/model-weights/stable-diffusion-v1-4", subfolder="unet")
print('Unet loaded')

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

prompt = ["a photograph of an astronaut riding a horse"]
neg_prompt = [" "]

height = 512
width = 512
num_inference_steps = 50
guidance_scale = 7.5
generator = torch.manual_seed(24)    # Seed generator to create the inital latent noise
batch_size = len(prompt)

# get the for the passed prompt. These embeddings will be used to condition the UNet model
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# get the unconditional text embeddings for classifier-free guidance
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# generate the initial random noise.
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

# initialize the scheduler with our chosen
scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

# denoise
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# normalize the image pixel from [-1, 1] to [0, 1]
image = (image / 2 + 0.5).clamp(0, 1)
# transport image to cpu and change the orders of tensor to adapt to numpy
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
# put pixel into the range of 0 to 255
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("output_image.png")
