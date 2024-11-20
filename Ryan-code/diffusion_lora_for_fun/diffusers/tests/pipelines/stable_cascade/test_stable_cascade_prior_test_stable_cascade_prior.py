def test_stable_cascade_prior(self):
    pipe = StableCascadePriorPipeline.from_pretrained(
        'stabilityai/stable-cascade-prior', variant='bf16', torch_dtype=
        torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    prompt = (
        'A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background.'
        )
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipe(prompt, num_inference_steps=2, output_type='np',
        generator=generator)
    image_embedding = output.image_embeddings
    expected_image_embedding = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_cascade/stable_cascade_prior_image_embeddings.npy'
        )
    assert image_embedding.shape == (1, 16, 24, 24)
    max_diff = numpy_cosine_similarity_distance(image_embedding.flatten(),
        expected_image_embedding.flatten())
    assert max_diff < 0.0001
