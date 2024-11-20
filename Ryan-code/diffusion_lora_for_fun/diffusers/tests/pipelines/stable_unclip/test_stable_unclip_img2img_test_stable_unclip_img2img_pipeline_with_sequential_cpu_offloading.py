def test_stable_unclip_img2img_pipeline_with_sequential_cpu_offloading(self):
    input_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png'
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        'fusing/stable-unclip-2-1-h-img2img', torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    _ = pipe(input_image, 'anime turtle', num_inference_steps=2,
        output_type='np')
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 7 * 10 ** 9
