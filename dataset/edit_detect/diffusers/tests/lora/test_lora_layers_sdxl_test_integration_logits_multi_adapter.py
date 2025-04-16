@nightly
def test_integration_logits_multi_adapter(self):
    path = 'stabilityai/stable-diffusion-xl-base-1.0'
    lora_id = 'CiroN2022/toy-face'
    pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=
        torch.float16)
    pipe.load_lora_weights(lora_id, weight_name='toy_face_sdxl.safetensors',
        adapter_name='toy')
    pipe = pipe.to(torch_device)
    self.assertTrue(check_if_lora_correctly_set(pipe.unet),
        'Lora not correctly set in Unet')
    prompt = 'toy_face of a hacker with a hoodie'
    lora_scale = 0.9
    images = pipe(prompt=prompt, num_inference_steps=30, generator=torch.
        manual_seed(0), cross_attention_kwargs={'scale': lora_scale},
        output_type='np').images
    expected_slice_scale = np.array([0.538, 0.539, 0.54, 0.54, 0.542, 0.539,
        0.538, 0.541, 0.539])
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
    pipe.load_lora_weights('nerijs/pixel-art-xl', weight_name=
        'pixel-art-xl.safetensors', adapter_name='pixel')
    pipe.set_adapters('pixel')
    prompt = 'pixel art, a hacker with a hoodie, simple, flat colors'
    images = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
        cross_attention_kwargs={'scale': lora_scale}, generator=torch.
        manual_seed(0), output_type='np').images
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    expected_slice_scale = np.array([0.61973065, 0.62018543, 0.62181497, 
        0.61933696, 0.6208608, 0.620576, 0.6200281, 0.62258327, 0.6259889])
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
    pipe.set_adapters(['pixel', 'toy'], adapter_weights=[0.5, 1.0])
    images = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
        cross_attention_kwargs={'scale': 1.0}, generator=torch.manual_seed(
        0), output_type='np').images
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    expected_slice_scale = np.array([0.5888, 0.5897, 0.5946, 0.5888, 0.5935,
        0.5946, 0.5857, 0.5891, 0.5909])
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
    pipe.disable_lora()
    images = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
        cross_attention_kwargs={'scale': lora_scale}, generator=torch.
        manual_seed(0), output_type='np').images
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    expected_slice_scale = np.array([0.5456, 0.5466, 0.5487, 0.5458, 0.5469,
        0.5454, 0.5446, 0.5479, 0.5487])
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
