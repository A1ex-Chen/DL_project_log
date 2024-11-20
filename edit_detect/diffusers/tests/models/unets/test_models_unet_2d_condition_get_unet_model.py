def get_unet_model(self, fp16=False, model_id='CompVis/stable-diffusion-v1-4'):
    revision = 'fp16' if fp16 else None
    torch_dtype = torch.float16 if fp16 else torch.float32
    model = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet',
        torch_dtype=torch_dtype, revision=revision)
    model.to(torch_device).eval()
    return model
