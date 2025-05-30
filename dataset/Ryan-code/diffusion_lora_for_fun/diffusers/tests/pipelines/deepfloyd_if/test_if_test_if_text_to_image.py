def test_if_text_to_image(self):
    pipe = IFPipeline.from_pretrained('DeepFloyd/IF-I-XL-v1.0', variant=
        'fp16', torch_dtype=torch.float16)
    pipe.unet.set_attn_processor(AttnAddedKVProcessor())
    pipe.enable_model_cpu_offload()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipe(prompt='anime turtle', num_inference_steps=2, generator=
        generator, output_type='np')
    image = output.images[0]
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 12 * 10 ** 9
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if.npy'
        )
    assert_mean_pixel_difference(image, expected_image)
    pipe.remove_all_hooks()
