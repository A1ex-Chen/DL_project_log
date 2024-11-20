def test_attend_and_excite_fp16(self):
    generator = torch.manual_seed(51)
    pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None, torch_dtype=
        torch.float16)
    pipe.to('cuda')
    prompt = 'a painting of an elephant with glasses'
    token_indices = [5, 7]
    image = pipe(prompt=prompt, token_indices=token_indices, guidance_scale
        =7.5, generator=generator, num_inference_steps=5, max_iter_to_alter
        =5, output_type='np').images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/attend-and-excite/elephant_glasses.npy'
        )
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        expected_image.flatten())
    assert max_diff < 0.5
