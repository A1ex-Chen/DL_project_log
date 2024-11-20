def test_stable_cascade_decoder(self):
    pipe = StableCascadeDecoderPipeline.from_pretrained(
        'stabilityai/stable-cascade', variant='bf16', torch_dtype=torch.
        bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    prompt = (
        'A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background.'
        )
    generator = torch.Generator(device='cpu').manual_seed(0)
    image_embedding = load_pt(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_cascade/image_embedding.pt'
        )
    image = pipe(prompt=prompt, image_embeddings=image_embedding,
        output_type='np', num_inference_steps=2, generator=generator).images[0]
    assert image.shape == (1024, 1024, 3)
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_cascade/stable_cascade_decoder_image.npy'
        )
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        expected_image.flatten())
    assert max_diff < 0.0001
