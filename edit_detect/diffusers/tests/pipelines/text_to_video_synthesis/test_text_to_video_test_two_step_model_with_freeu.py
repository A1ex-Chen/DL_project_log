def test_two_step_model_with_freeu(self):
    expected_video = []
    pipe = TextToVideoSDPipeline.from_pretrained(
        'damo-vilab/text-to-video-ms-1.7b')
    pipe = pipe.to(torch_device)
    prompt = 'Spiderman is surfing'
    generator = torch.Generator(device='cpu').manual_seed(0)
    pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    video_frames = pipe(prompt, generator=generator, num_inference_steps=2,
        output_type='np').frames
    video = video_frames[0, 0, -3:, -3:, -1].flatten()
    expected_video = [0.3643, 0.3455, 0.3831, 0.3923, 0.2978, 0.3247, 
        0.3278, 0.3201, 0.3475]
    assert np.abs(expected_video - video).mean() < 0.05
