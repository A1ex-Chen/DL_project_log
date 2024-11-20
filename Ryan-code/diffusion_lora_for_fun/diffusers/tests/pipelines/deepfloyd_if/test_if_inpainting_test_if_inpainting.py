def test_if_inpainting(self):
    pipe = IFInpaintingPipeline.from_pretrained('DeepFloyd/IF-I-XL-v1.0',
        variant='fp16', torch_dtype=torch.float16)
    pipe.unet.set_attn_processor(AttnAddedKVProcessor())
    pipe.enable_model_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device
        )
    mask_image = floats_tensor((1, 3, 64, 64), rng=random.Random(1)).to(
        torch_device)
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipe(prompt='anime prompts', image=image, mask_image=
        mask_image, num_inference_steps=2, generator=generator, output_type
        ='np')
    image = output.images[0]
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 12 * 10 ** 9
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_inpainting.npy'
        )
    assert_mean_pixel_difference(image, expected_image)
    pipe.remove_all_hooks()
