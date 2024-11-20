def test_stable_diffusion_textual_inversion_with_sequential_cpu_offload(self):
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4')
    pipe.enable_sequential_cpu_offload()
    pipe.load_textual_inversion('sd-concepts-library/low-poly-hd-logos-icons')
    a111_file = hf_hub_download(
        'hf-internal-testing/text_inv_embedding_a1111_format',
        'winter_style.pt')
    a111_file_neg = hf_hub_download(
        'hf-internal-testing/text_inv_embedding_a1111_format',
        'winter_style_negative.pt')
    pipe.load_textual_inversion(a111_file)
    pipe.load_textual_inversion(a111_file_neg)
    generator = torch.Generator(device='cpu').manual_seed(1)
    prompt = (
        'An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>'
        )
    neg_prompt = 'Style-Winter-neg'
    image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=
        generator, output_type='np').images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.8
