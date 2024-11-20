@nightly
def test_sequential_fuse_unfuse(self):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
    pipe.load_lora_weights('Pclanglais/TintinIA', torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.fuse_lora()
    generator = torch.Generator().manual_seed(0)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    image_slice = images[0, -3:, -3:, -1].flatten()
    pipe.unfuse_lora()
    pipe.load_lora_weights('ProomptEngineer/pe-balloon-diffusion-style',
        torch_dtype=torch.float16)
    pipe.fuse_lora()
    pipe.unfuse_lora()
    pipe.load_lora_weights('ostris/crayon_style_lora_sdxl', torch_dtype=
        torch.float16)
    pipe.fuse_lora()
    pipe.unfuse_lora()
    pipe.load_lora_weights('Pclanglais/TintinIA', torch_dtype=torch.float16)
    pipe.fuse_lora()
    generator = torch.Generator().manual_seed(0)
    images_2 = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    image_slice_2 = images_2[0, -3:, -3:, -1].flatten()
    max_diff = numpy_cosine_similarity_distance(image_slice, image_slice_2)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
