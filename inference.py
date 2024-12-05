from PIL import Image
import os
import requests
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.show()
from tqdm import tqdm

import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline, 
    StableDiffusionPipeline,
    StableDiffusionInpaintPipelineLegacy, 
    DDIMScheduler, 
    AutoencoderKL,
    StableVideoDiffusionPipeline,
    I2VGenXLPipeline
)
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from ip_adapter import IPAdapter

from utils import *

past_me_path = Path("/root/FaceSynthesis/images/past_me.jpg")
current_me_path = Path("/root/FaceSynthesis/images/current_me.jpg")
future_me_path = Path("/root/FaceSynthesis/images/future_me.jpg")
scene_path = Path("/root/FaceSynthesis/images/scene_gta.jpg")
if "gta" in str(scene_path):
    mask_path_pt = Path("/root/FaceSynthesis/masks_gta.pt")
elif "spurs" in str(scene_path):
    mask_path_pt = Path("/root/FaceSynthesis/masks_spurs.pt")
# mask_path_pt = Path("/root/FaceSynthesis/masks.pt")
result_image_path = Path("/root/FaceSynthesis/images/inpaint_result.jpg")

# For reproduction, please download and change the paths to ip checkpoint and image encoder
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = Path("/root/autodl-tmp/model_weights/IPAdapter/models/image_encoder")
ip_ckpt = Path("/root/autodl-tmp/model_weights/IPAdapter/models/ip-adapter_sd15.bin") 

GEN_FROM_SCRATCH = False # Please switch to True if you want to reproduce the results from scratch

# Generate future me image using InstructPix2Pix
def generate_future_me(current_me_path, future_me_path):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix")
    pipe.to(device)
    prompt = "Make this man look like in his 80s."
    current_me_image = Image.open(current_me_path)
    image = pipe(prompt=prompt, image=current_me_image).images[0]
    image.save(future_me_path)
    del pipe

# Replace faces in the scene with me images
def generate_face_mask_in_scene(scene_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model.to(device)
    
    # Generate masks with SAM
    '''
    GTA picture:
    Trevor: 327, 314, 511, 542
    Franklin: 1115,298,1302,529
    Michael: 1926, 298, 2118, 536

    Spurs picture:
    Duncan: 356, 128, 492, 321
    Manu:  525, 195, 647, 367
    Parker: 703, 235, 839, 425
    '''
    raw_image = Image.open(scene_path).convert("RGB")

    if "gta" in str(scene_path):
        input_boxes_lst = [[327, 314, 511, 542], [1115,298,1302,529], [1926, 298, 2118, 536]]
        input_points_lst = [[416, 418], [1218, 376], [2044, 412]]
        scene_name = "gta"
    elif "spurs" in str(scene_path):
        input_boxes_lst = [[356, 128, 492, 321], [525, 195, 647, 367], [703, 235, 839, 425]]
        input_points_lst = [[436, 243], [591, 283], [755, 336]]
        scene_name = "spurs"

    # input_boxes_lst = [[356, 128, 492, 321], [525, 195, 647, 367], [703, 235, 839, 425]] # Suprs
    # input_boxes_lst = [[327, 314, 511, 542], [1115,298,1302,529], [1926, 298, 2118, 536]] # GTA
    mask_lst = []

    for input_box, input_point in tqdm(zip(input_boxes_lst, input_points_lst), desc="Generating masks with SAM"):
        input_boxes = [[input_box]]
        input_points = [[input_point]]
        inputs = processor(
            raw_image, input_boxes=[input_boxes], 
            input_points=[input_points],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores
        show_masks_on_image(raw_image, masks[0], scores) # See utils.py
        mask_lst.append(masks[0])
    
    # Save the masks to pt file
    tensor_dict = {}
    for i, mask in enumerate(mask_lst):
        tensor_dict["mask_" + str(i)] = mask
    torch.save(tensor_dict, mask_path_pt)
    del model
    return mask_lst

def inpaint_faces(
    scene_path,
    past_me_path,
    current_me_path,
    future_me_path,
    masks_path
):
    # Load the masks from pt and save to png
    masks = torch.load(masks_path)
    for i, mask_batch in masks.items():
        for j, mask in enumerate(mask_batch):
            save_mask_to_png(mask, f"/root/FaceSynthesis/images/mask_{i}_{j}.png") # See utils.py
    
    # Load ip-adapter
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    scene_image = Image.open(scene_path).convert("RGB")
    if "gta" in str(scene_path):
        max_size = int(max(scene_image.size) / 2)
        scene_image = resize_image(scene_path, max_size=max_size)

    # Inpaint current me
    image_current_me = Image.open(current_me_path).convert("RGB")
    image_current_me = image_current_me
    mask_current_me = Image.open("/root/FaceSynthesis/images/mask_mask_1_0.png").convert("L")
    if "gta" in str(scene_path):
        mask_current_me = mask_current_me.resize(scene_image.size)
    images_gen_current_me = ip_model.generate(
        pil_image=image_current_me,
        num_samples=4,
        num_inference_steps=60,
        seed=42,
        image=scene_image,
        mask_image=mask_current_me,
        strength=0.6
    )
    grid = image_grid(images_gen_current_me, 4, 1) # see utils
    grid.save("/root/FaceSynthesis/images/inpaint_current_me_grid.jpg") # Pick the 4th image

    # Inpaint future me
    scene_image_current = images_gen_current_me[3]
    image_future_me = Image.open(future_me_path).convert("RGB")
    mask_future_me = Image.open("/root/FaceSynthesis/images/mask_mask_2_0.png").convert("L")
    if "gta" in str(scene_path):
        mask_future_me = mask_future_me.resize(scene_image_current.size)
    images_gen_future_me = ip_model.generate(
        pil_image=image_future_me,
        num_samples=4,
        num_inference_steps=60,
        seed=42,
        image=scene_image_current,
        mask_image=mask_future_me,
        strength=0.6
    )
    grid = image_grid(images_gen_future_me, 4, 1) # see utils
    grid.save("/root/FaceSynthesis/images/inpaint_future_me_grid.jpg") # Pick the 4th image

    # Inpaint past me
    scene_image_future = images_gen_future_me[3]
    image_past_me = Image.open(past_me_path).convert("RGB")
    mask_past_me = Image.open("/root/FaceSynthesis/images/mask_mask_0_0.png").convert("L")
    if "gta" in str(scene_path):
        mask_past_me = mask_past_me.resize(scene_image_future.size)
    images_gen_past_me = ip_model.generate(
        pil_image=image_past_me,
        num_samples=4,
        num_inference_steps=60,
        seed=42,
        image=scene_image_future,
        mask_image=mask_past_me,
        strength=0.6
    )
    grid = image_grid(images_gen_past_me, 4, 1) # see utils
    grid.save("/root/FaceSynthesis/images/inpaint_past_me_grid.jpg") # Pick the first image
    del ip_model
    del pipe
    del vae

    res_image = images_gen_past_me[0]
    res_image.save("/root/FaceSynthesis/images/inpaint_result.jpg")

def generate_video(
    image_path,
    prompt
):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    # pipeline.enable_model_cpu_offload()
    pipeline.to(device)

    image = load_image(str(image_path))
    # image = image.resize((1024, 576))

    generator = torch.manual_seed(42)
    frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
    export_to_video(frames, "/root/FaceSynthesis/generated.mp4", fps=7)
    # pipeline = I2VGenXLPipeline.from_pretrained(
    #     "ali-vilab/i2vgen-xl", 
    #     torch_dtype=torch.float16, 
    #     variant="fp16"
    # )
    # pipeline.to(device)

    # image = load_image(str(image_path))
    # # prompt = "Three men hugging each other in a basketball court."
    # negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    # generator = torch.manual_seed(42)

    # frames = pipeline(
    #     prompt=prompt,
    #     image=image,
    #     num_inference_steps=80,
    #     negative_prompt=negative_prompt,
    #     guidance_scale=9.0,
    #     generator=generator
    # ).frames[0]
    # export_to_video(frames, "/root/FaceSynthesis/generated.mp4", fps=5)

def main():
    if GEN_FROM_SCRATCH:
        generate_future_me(current_me_path, future_me_path)
        print("Future me image generated.")

        mask_lst = generate_face_mask_in_scene(scene_path, current_me_path, future_me_path)
        print("Masks generated.")

        inpaint_faces(
            scene_path,
            past_me_path,
            current_me_path,
            future_me_path,
            mask_path_pt
        )
        print("Inpainting done.")
    else:
        if not os.path.exists(future_me_path):
            generate_future_me(current_me_path, future_me_path)
            print("Future me image generated.")
        else:
            print("Future me image already exists.")
        
        scene_name = "spurs" if "spurs" in str(scene_path) else "gta"

        if not os.path.exists(mask_path_pt):
            mask_lst = generate_face_mask_in_scene(scene_path)
            print("Masks generated.")
        else:
            print("Masks already exist.")

        if not os.path.exists(result_image_path):
            inpaint_faces(
                scene_path,
                past_me_path,
                current_me_path,
                future_me_path,
                mask_path_pt
            )
            print("Inpainting done.")
        else:
            print("Inpainting already done.")
        
        generate_video(
            result_image_path, 
            prompt = "Three men looking at each other and smile."
        )
        

if __name__ == "__main__":
    main()