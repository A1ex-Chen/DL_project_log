def build_stable_diffusion_model(self):
    self.base = StableDiffusionXLPipeline.from_pretrained(sd_model_base,
        torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
    self.base.to(self.device)
    self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        sd_model_refiner, text_encoder_2=self.base.text_encoder_2, vae=self
        .base.vae, torch_dtype=torch.float16, use_safetensors=True, variant
        ='fp16')
    self.refiner.to(self.device)
    self.edit_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        sd_model_edit, torch_dtype=torch.float16, variant='fp16',
        use_safetensors=True)
    self.edit_model.to(self.device)
    self.generator = torch.Generator('cuda').manual_seed(0)
