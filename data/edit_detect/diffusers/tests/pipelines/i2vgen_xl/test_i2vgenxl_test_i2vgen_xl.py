def test_i2vgen_xl(self):
    pipe = I2VGenXLPipeline.from_pretrained('ali-vilab/i2vgen-xl',
        torch_dtype=torch.float16, variant='fp16')
    pipe = pipe.to(torch_device)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true'
        )
    generator = torch.Generator('cpu').manual_seed(0)
    num_frames = 3
    output = pipe(image=image, prompt='my cat', num_frames=num_frames,
        generator=generator, num_inference_steps=3, output_type='np')
    image = output.frames[0]
    assert image.shape == (num_frames, 704, 1280, 3)
    image_slice = image[0, -3:, -3:, -1]
    print_tensor_test(image_slice.flatten())
    expected_slice = np.array([0.5482, 0.6244, 0.6274, 0.4584, 0.5935, 
        0.5937, 0.4579, 0.5767, 0.5892])
    assert numpy_cosine_similarity_distance(image_slice.flatten(),
        expected_slice.flatten()) < 0.001
