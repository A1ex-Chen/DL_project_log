@require_torch_gpu
def test_integration_move_lora_dora_cpu(self):
    from peft import LoraConfig
    path = 'runwayml/stable-diffusion-v1-5'
    unet_lora_config = LoraConfig(init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'], use_dora=True)
    text_lora_config = LoraConfig(init_lora_weights='gaussian',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'], use_dora
        =True)
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.
        float16)
    pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
    pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
    self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
        'Lora not correctly set in text encoder')
    self.assertTrue(check_if_lora_correctly_set(pipe.unet),
        'Lora not correctly set in text encoder')
    for name, param in pipe.unet.named_parameters():
        if 'lora_' in name:
            self.assertEqual(param.device, torch.device('cpu'))
    for name, param in pipe.text_encoder.named_parameters():
        if 'lora_' in name:
            self.assertEqual(param.device, torch.device('cpu'))
    pipe.set_lora_device(['adapter-1'], torch_device)
    for name, param in pipe.unet.named_parameters():
        if 'lora_' in name:
            self.assertNotEqual(param.device, torch.device('cpu'))
    for name, param in pipe.text_encoder.named_parameters():
        if 'lora_' in name:
            self.assertNotEqual(param.device, torch.device('cpu'))
