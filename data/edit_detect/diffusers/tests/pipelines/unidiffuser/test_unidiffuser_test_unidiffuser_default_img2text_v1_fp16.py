def test_unidiffuser_default_img2text_v1_fp16(self):
    pipe = UniDiffuserPipeline.from_pretrained('thu-ml/unidiffuser-v1',
        torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(device=torch_device, generate_latents=True)
    del inputs['prompt']
    sample = pipe(**inputs)
    text = sample.text
    expected_text_prefix = 'An astronaut'
    assert text[0][:len(expected_text_prefix)] == expected_text_prefix
