from PIL import Image
import os
import requests
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.show()
from tqdm import tqdm

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from utils import *


current_me_path = Path("/root/FaceSynthesis/images/current_me.jpg")
future_me_path = Path("/root/FaceSynthesis/images/future_me.jpg")
scene_path = Path("/root/FaceSynthesis/images/scene.jpg")

# Generate future me image using InstructPix2Pix
def generate_future_me(current_me_path, future_me_path):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix")
    pipe.to("cuda")
    prompt = "Make this man look like in his 80s."
    current_me_image = Image.open(current_me_path)
    image = pipe(prompt=prompt, image=current_me_image).images[0]
    image.save(future_me_path)

# Replace faces in the scene with me images
def replace_faces_in_scene(scene_path, current_me_path, future_me_path):
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

    input_boxes_lst = [[1115, 298, 1302, 529], [1926, 298, 2118, 536], [327, 314, 511, 542]] # GTA, sorted by age
    mask_lst = []

    for input_box in tqdm(input_boxes_lst, desc="Generating masks with SAM"):
        input_boxes = [[input_box]]
        inputs = processor(
            raw_image, input_boxes=[input_boxes], 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores
        show_masks_on_image(raw_image, masks[0], scores)
        mask_lst.append(masks[0])
    
    # Save the masks
    tensor_dict = {}
    for i, mask in enumerate(mask_lst):
        tensor_dict["mask_" + str(i)] = mask
    torch.save(tensor_dict, "/root/FaceSynthesis/masks.pt")
    return mask_lst

# 主函数
def main():
    if not os.path.exists(future_me_path):
        generate_future_me(current_me_path, future_me_path)
        print("Future me image generated.")
    else:
        print("Future me image already exists.")
    if not os.path.exists("/root/FaceSynthesis/masks.pt"):
        mask_lst = replace_faces_in_scene(scene_path, current_me_path, future_me_path)
        print("Masks generated.")
    else:
        print("Masks already exist.")

    # Inpainting
    

if __name__ == "__main__":
    main()
