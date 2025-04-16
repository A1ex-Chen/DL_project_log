# https://github.com/huggingface/diffusers/issues/4654

from diffusers import DDIMScheduler, UNet2DModel, DDIMInverseScheduler
import torch
import numpy as np
import cv2
from tqdm import tqdm






scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
invscheduler = DDIMInverseScheduler.from_pretrained("google/ddpm-cifar10-32")
model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True).to("cuda")


noise = np.random.randn(1, 3, 32, 32).astype("float32")
x0 = ddpm_forward(noise, model, scheduler)
oimg = (x0 / 2 + 0.5).clamp(0, 1).squeeze()
oimg = (oimg.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
cv2.imwrite("ddim_gen.jpg", oimg)
    


latent = ddpm_backward(x0, model, invscheduler)

nimg = ddpm_forward(latent, model, scheduler)
nimg = (nimg / 2 + 0.5).clamp(0, 1).squeeze()
nimg = (nimg.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
cv2.imwrite("ddim_re.jpg", nimg)

err = oimg/255 - nimg/255
import pdb;pdb.set_trace()