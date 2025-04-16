def test_two_step_model(self):
    pipe = VideoToVideoSDPipeline.from_pretrained('cerspense/zeroscope_v2_576w'
        , torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    video = torch.randn((1, 10, 3, 320, 576), generator=generator)
    prompt = 'Spiderman is surfing'
    generator = torch.Generator(device='cpu').manual_seed(0)
    video_frames = pipe(prompt, video=video, generator=generator,
        num_inference_steps=3, output_type='np').frames
    expected_array = np.array([0.17114258, 0.13720703, 0.08886719, 
        0.14819336, 0.1730957, 0.24584961, 0.22021484, 0.35180664, 0.2607422])
    output_array = video_frames[0, 0, :3, :3, 0].flatten()
    assert numpy_cosine_similarity_distance(expected_array, output_array
        ) < 0.001
