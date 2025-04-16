def test_local_save_load_index(self):
    prompt = 'hello'
    for variant in [None, 'fp16']:
        for use_safe in [True, False]:
            pipe = StableDiffusionPipeline.from_pretrained(
                'hf-internal-testing/tiny-stable-diffusion-pipe-indexes',
                variant=variant, use_safetensors=use_safe, safety_checker=None)
            pipe = pipe.to(torch_device)
            generator = torch.manual_seed(0)
            out = pipe(prompt, num_inference_steps=2, generator=generator,
                output_type='np').images
            with tempfile.TemporaryDirectory() as tmpdirname:
                pipe.save_pretrained(tmpdirname)
                pipe_2 = StableDiffusionPipeline.from_pretrained(tmpdirname,
                    safe_serialization=use_safe, variant=variant)
                pipe_2 = pipe_2.to(torch_device)
            generator = torch.manual_seed(0)
            out_2 = pipe_2(prompt, num_inference_steps=2, generator=
                generator, output_type='np').images
            assert np.max(np.abs(out - out_2)) < 0.001
