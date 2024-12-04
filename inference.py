from PIL import Image
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
plt.show()

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline

current_me_path = "images/current_me.jpg"
future_me_path = "images/future_me.jpg"
scene_path = "images/scene.jpg"

# Generate future me image using InstructPix2Pix
def generate_future_me(current_me_path, future_me_path):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix")
    pipe.to("cuda")
    prompt = "Make this man look like in his 80s."
    current_me_image = Image.open(current_me_path)
    image = pipe(prompt=prompt, image=current_me_image).images[0]
    image.save(future_me_path)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()
    plt.savefig("images/masks.jpg")

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
    '''
    raw_image = Image.open(scene_path).convert("RGB")
    input_boxes = [[[1115,298,1302,529]]]  # 2D location of a window in the image
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
    
    # Inpaint
    # inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    # inpaint_pipe.to("cuda")
    # inpainted_image = inpaint_pipe(prompt="Replace with my face", image=scene_path, mask_image=masks).images[0]
    # inpainted_image.save(scene_path)

# 主函数
def main():
    if not os.path.exists(current_me_path):
        generate_future_me(current_me_path, future_me_path)
        print("Future me image generated.")
    else:
        print("Future me image already exists.")
    replace_faces_in_scene(scene_path, current_me_path, future_me_path)

if __name__ == "__main__":
    main()
