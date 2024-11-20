def test_sdxl_1_0_fuse_unfuse_all(self):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
    text_encoder_1_sd = copy.deepcopy(pipe.text_encoder.state_dict())
    text_encoder_2_sd = copy.deepcopy(pipe.text_encoder_2.state_dict())
    unet_sd = copy.deepcopy(pipe.unet.state_dict())
    pipe.load_lora_weights('davizca87/sun-flower', weight_name=
        'snfw3rXL-000004.safetensors', torch_dtype=torch.float16)
    fused_te_state_dict = pipe.text_encoder.state_dict()
    fused_te_2_state_dict = pipe.text_encoder_2.state_dict()
    unet_state_dict = pipe.unet.state_dict()
    peft_ge_070 = version.parse(importlib.metadata.version('peft')
        ) >= version.parse('0.7.0')

    def remap_key(key, sd):
        if key in sd or not peft_ge_070:
            return key
        if key.endswith('.weight'):
            key = key[:-7] + '.base_layer.weight'
        elif key.endswith('.bias'):
            key = key[:-5] + '.base_layer.bias'
        return key
    for key, value in text_encoder_1_sd.items():
        key = remap_key(key, fused_te_state_dict)
        self.assertTrue(torch.allclose(fused_te_state_dict[key], value))
    for key, value in text_encoder_2_sd.items():
        key = remap_key(key, fused_te_2_state_dict)
        self.assertTrue(torch.allclose(fused_te_2_state_dict[key], value))
    for key, value in unet_state_dict.items():
        self.assertTrue(torch.allclose(unet_state_dict[key], value))
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    assert not state_dicts_almost_equal(text_encoder_1_sd, pipe.
        text_encoder.state_dict())
    assert not state_dicts_almost_equal(text_encoder_2_sd, pipe.
        text_encoder_2.state_dict())
    assert not state_dicts_almost_equal(unet_sd, pipe.unet.state_dict())
    release_memory(pipe)
    del unet_sd, text_encoder_1_sd, text_encoder_2_sd
