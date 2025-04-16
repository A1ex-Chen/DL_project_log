def test_not_empty_state_dict(self):
    pipe = AutoPipelineForText2Image.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to(
        torch_device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    cached_file = hf_hub_download('hf-internal-testing/lcm-lora-test-sd-v1-5',
        'test_lora.safetensors')
    lcm_lora = load_file(cached_file)
    pipe.load_lora_weights(lcm_lora, adapter_name='lcm')
    self.assertTrue(lcm_lora != {})
    release_memory(pipe)
