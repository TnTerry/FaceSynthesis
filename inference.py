import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
from transformers import CLIPVisionModelWithProjection
from segment_anything import SamPredictor, sam_model_registry
import cv2

class FaceSynthesisPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        # 初始化IP-Adapter模型
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        ).to(device)
        
        # 初始化SAM模型用于人物分割
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(self.sam)
        
        # 加载IP-Adapter权重
        self.load_ip_adapter()

    def load_ip_adapter(self):
        """加载IP-Adapter权重和配置"""
        ip_ckpt = torch.load("ip-adapter_sd15.bin", map_location=self.device)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device, dtype=torch.float16)
        
        # 应用IP-Adapter权重到UNet
        ip_layers = torch.nn.ModuleList([
            torch.nn.Linear(768, self.pipe.unet.config.cross_attention_dim)
            for _ in range(12)
        ]).to(self.device, dtype=torch.float16)
        ip_layers.load_state_dict(ip_ckpt)
        self.pipe.unet.set_ip_adapter(ip_layers)

    def segment_person(self, image):
        """使用SAM进行人物分割"""
        image_array = np.array(image)
        self.sam_predictor.set_image(image_array)
        
        # 使用图像中心点作为提示
        h, w = image_array.shape[:2]
        center_point = np.array([[w//2, h//2]])
        masks, _, _ = self.sam_predictor.predict(
            point_coords=center_point,
            point_labels=np.array([1]),
            multimask_output=False
        )
        return Image.fromarray(masks[0].astype(np.uint8) * 255)

    def process_reference_image(self, image):
        """处理参考图像，提取特征"""
        image = image.resize((224, 224))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, dtype=torch.float16) / 255.0
        return self.image_encoder(image).image_embeds

    def generate_composite(self, base_image, reference_image, prompt, strength=0.8):
        """生成合成图像"""
        # 处理参考图像
        image_embeds = self.process_reference_image(reference_image)
        
        # 生成分割蒙版
        mask = self.segment_person(base_image)
        
        # 准备生成参数
        num_inference_steps = 50
        guidance_scale = 7.5
        
        # 使用IP-Adapter生成图像
        image = self.pipe(
            prompt=prompt,
            image=base_image,
            mask_image=mask,
            image_embeds=image_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
        
        return image

    def save_result(self, image, path):
        """保存生成的图像"""
        image.save(path)

def main():
    # 使用示例
    pipeline = FaceSynthesisPipeline()
    
    # 加载图像
    base_image = Image.open("/images/base_scene.jpg")
    reference_image = Image.open("/images/reference_face.jpg")
    
    # 生成合成图像
    prompt = "A person standing in a natural scene, high quality, detailed face"
    result = pipeline.generate_composite(base_image, reference_image, prompt)
    
    # 保存结果
    pipeline.save_result(result, "output.png")

if __name__ == "__main__":
    main()