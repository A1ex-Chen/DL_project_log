def test_dit_512(self):
    pipe = DiTPipeline.from_pretrained('facebook/DiT-XL-2-512')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler
        .config)
    pipe.to('cuda')
    words = ['vase', 'umbrella']
    ids = pipe.get_label_ids(words)
    generator = torch.manual_seed(0)
    images = pipe(ids, generator=generator, num_inference_steps=25,
        output_type='np').images
    for word, image in zip(words, images):
        expected_image = load_numpy(
            f'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/{word}_512.npy'
            )
        assert np.abs((expected_image - image).max()) < 0.1
