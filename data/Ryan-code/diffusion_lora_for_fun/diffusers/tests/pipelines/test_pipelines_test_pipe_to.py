def test_pipe_to(self):
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    device_type = torch.device(torch_device).type
    sd1 = sd.to(device_type)
    sd2 = sd.to(torch.device(device_type))
    sd3 = sd.to(device_type, torch.float32)
    sd4 = sd.to(device=device_type)
    sd5 = sd.to(torch_device=device_type)
    sd6 = sd.to(device_type, dtype=torch.float32)
    sd7 = sd.to(device_type, torch_dtype=torch.float32)
    assert sd1.device.type == device_type
    assert sd2.device.type == device_type
    assert sd3.device.type == device_type
    assert sd4.device.type == device_type
    assert sd5.device.type == device_type
    assert sd6.device.type == device_type
    assert sd7.device.type == device_type
    sd1 = sd.to(torch.float16)
    sd2 = sd.to(None, torch.float16)
    sd3 = sd.to(dtype=torch.float16)
    sd4 = sd.to(dtype=torch.float16)
    sd5 = sd.to(None, dtype=torch.float16)
    sd6 = sd.to(None, torch_dtype=torch.float16)
    assert sd1.dtype == torch.float16
    assert sd2.dtype == torch.float16
    assert sd3.dtype == torch.float16
    assert sd4.dtype == torch.float16
    assert sd5.dtype == torch.float16
    assert sd6.dtype == torch.float16
    sd1 = sd.to(device=device_type, dtype=torch.float16)
    sd2 = sd.to(torch_device=device_type, torch_dtype=torch.float16)
    sd3 = sd.to(device_type, torch.float16)
    assert sd1.dtype == torch.float16
    assert sd2.dtype == torch.float16
    assert sd3.dtype == torch.float16
    assert sd1.device.type == device_type
    assert sd2.device.type == device_type
    assert sd3.device.type == device_type
