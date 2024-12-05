import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Source: SAM Demo notebook
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks_on_image(
  raw_image, 
  masks, 
  scores, 
  path = "/root/FaceSynthesis/images/masks.jpg"
):
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
    plt.savefig(path)


# Save SAM mask to PNG
def save_mask_to_png(mask_tensor, save_path):
  if mask_tensor.is_cuda:
      mask_tensor = mask_tensor.cpu()
  mask_numpy = mask_tensor.numpy()
  
  mask_numpy = mask_numpy[0]
  
  mask_uint8 = (mask_numpy * 255).astype(np.uint8)
  mask_image = Image.fromarray(mask_uint8, mode='L')
  mask_image.save(save_path)

# Copied from IP-Adapter repo: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_demo.ipynb
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def resize_image(image_path, output_path=None, max_size=512):
  img = Image.open(image_path)
  ratio = min(max_size/img.width, max_size/img.height)
  new_size = (int(img.width*ratio), int(img.height*ratio))
  resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
  if output_path:
    resized_img.save(output_path)
  return resized_img