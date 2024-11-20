from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("C:/Users/ryan/Documents/LoraOutput/pytorch_lora_weights.safetensors", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A wieght lion is running").images[0]
image.save("output_image.png")