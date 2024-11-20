@parameterized.expand([[DDIMScheduler, DDIMPipeline, 32], [DDPMScheduler,
    DDPMPipeline, 32], [DDIMScheduler, DDIMPipeline, (32, 64)], [
    DDPMScheduler, DDPMPipeline, (64, 32)]])
def test_uncond_unet_components(self, scheduler_fn=DDPMScheduler,
    pipeline_fn=DDPMPipeline, sample_size=32):
    unet = self.dummy_uncond_unet(sample_size)
    scheduler = scheduler_fn()
    pipeline = pipeline_fn(unet, scheduler).to(torch_device)
    generator = torch.manual_seed(0)
    out_image = pipeline(generator=generator, num_inference_steps=2,
        output_type='np').images
    sample_size = (sample_size, sample_size) if isinstance(sample_size, int
        ) else sample_size
    assert out_image.shape == (1, *sample_size, 3)
