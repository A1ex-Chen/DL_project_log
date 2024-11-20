def test_two_step_model(self):
    expected_video = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text-to-video/video_2step.npy'
        )
    pipe = TextToVideoSDPipeline.from_pretrained(
        'damo-vilab/text-to-video-ms-1.7b')
    pipe = pipe.to(torch_device)
    prompt = 'Spiderman is surfing'
    generator = torch.Generator(device='cpu').manual_seed(0)
    video_frames = pipe(prompt, generator=generator, num_inference_steps=2,
        output_type='np').frames
    assert numpy_cosine_similarity_distance(expected_video.flatten(),
        video_frames.flatten()) < 0.0001
