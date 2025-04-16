def test_sd_video(self):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid', variant='fp16',
        torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true'
        )
    generator = torch.Generator('cpu').manual_seed(0)
    num_frames = 3
    output = pipe(image=image, num_frames=num_frames, generator=generator,
        num_inference_steps=3, output_type='np')
    image = output.frames[0]
    assert image.shape == (num_frames, 576, 1024, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.8592, 0.8645, 0.8499, 0.8722, 0.8769, 
        0.8421, 0.8557, 0.8528, 0.8285])
    assert numpy_cosine_similarity_distance(image_slice.flatten(),
        expected_slice.flatten()) < 0.001
