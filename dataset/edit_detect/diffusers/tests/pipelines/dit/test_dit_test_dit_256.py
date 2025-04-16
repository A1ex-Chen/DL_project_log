def test_dit_256(self):
    generator = torch.manual_seed(0)
    pipe = DiTPipeline.from_pretrained('facebook/DiT-XL-2-256')
    pipe.to('cuda')
    words = ['vase', 'umbrella', 'white shark', 'white wolf']
    ids = pipe.get_label_ids(words)
    images = pipe(ids, generator=generator, num_inference_steps=40,
        output_type='np').images
    for word, image in zip(words, images):
        expected_image = load_numpy(
            f'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/{word}.npy'
            )
        assert np.abs((expected_image - image).max()) < 0.01
