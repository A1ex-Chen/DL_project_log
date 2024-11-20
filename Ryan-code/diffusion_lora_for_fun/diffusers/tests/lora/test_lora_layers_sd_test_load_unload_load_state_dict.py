def test_load_unload_load_state_dict(self):
    pipe = AutoPipelineForText2Image.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to(
        torch_device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    cached_file = hf_hub_download('hf-internal-testing/lcm-lora-test-sd-v1-5',
        'test_lora.safetensors')
    lcm_lora = load_file(cached_file)
    previous_state_dict = lcm_lora.copy()
    pipe.load_lora_weights(lcm_lora, adapter_name='lcm')
    self.assertDictEqual(lcm_lora, previous_state_dict)
    pipe.unload_lora_weights()
    pipe.load_lora_weights(lcm_lora, adapter_name='lcm')
    self.assertDictEqual(lcm_lora, previous_state_dict)
    release_memory(pipe)
