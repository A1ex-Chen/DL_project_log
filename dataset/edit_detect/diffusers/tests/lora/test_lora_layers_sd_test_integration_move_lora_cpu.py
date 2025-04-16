@slow
@require_torch_gpu
def test_integration_move_lora_cpu(self):
    path = 'runwayml/stable-diffusion-v1-5'
    lora_id = 'takuma104/lora-test-text-encoder-lora-target'
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.
        float16)
    pipe.load_lora_weights(lora_id, adapter_name='adapter-1')
    pipe.load_lora_weights(lora_id, adapter_name='adapter-2')
    pipe = pipe.to(torch_device)
    self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
        'Lora not correctly set in text encoder')
    self.assertTrue(check_if_lora_correctly_set(pipe.unet),
        'Lora not correctly set in text encoder')
    pipe.set_lora_device(['adapter-1'], 'cpu')
    for name, module in pipe.unet.named_modules():
        if 'adapter-1' in name and not isinstance(module, (nn.Dropout, nn.
            Identity)):
            self.assertTrue(module.weight.device == torch.device('cpu'))
        elif 'adapter-2' in name and not isinstance(module, (nn.Dropout, nn
            .Identity)):
            self.assertTrue(module.weight.device != torch.device('cpu'))
    for name, module in pipe.text_encoder.named_modules():
        if 'adapter-1' in name and not isinstance(module, (nn.Dropout, nn.
            Identity)):
            self.assertTrue(module.weight.device == torch.device('cpu'))
        elif 'adapter-2' in name and not isinstance(module, (nn.Dropout, nn
            .Identity)):
            self.assertTrue(module.weight.device != torch.device('cpu'))
    pipe.set_lora_device(['adapter-1'], 0)
    for n, m in pipe.unet.named_modules():
        if 'adapter-1' in n and not isinstance(m, (nn.Dropout, nn.Identity)):
            self.assertTrue(m.weight.device != torch.device('cpu'))
    for n, m in pipe.text_encoder.named_modules():
        if 'adapter-1' in n and not isinstance(m, (nn.Dropout, nn.Identity)):
            self.assertTrue(m.weight.device != torch.device('cpu'))
    pipe.set_lora_device(['adapter-1', 'adapter-2'], torch_device)
    for n, m in pipe.unet.named_modules():
        if ('adapter-1' in n or 'adapter-2' in n) and not isinstance(m, (nn
            .Dropout, nn.Identity)):
            self.assertTrue(m.weight.device != torch.device('cpu'))
    for n, m in pipe.text_encoder.named_modules():
        if ('adapter-1' in n or 'adapter-2' in n) and not isinstance(m, (nn
            .Dropout, nn.Identity)):
            self.assertTrue(m.weight.device != torch.device('cpu'))
