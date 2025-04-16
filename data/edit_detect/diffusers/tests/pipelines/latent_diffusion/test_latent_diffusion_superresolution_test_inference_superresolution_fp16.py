@unittest.skipIf(torch_device != 'cuda', 'This test requires a GPU')
def test_inference_superresolution_fp16(self):
    unet = self.dummy_uncond_unet
    scheduler = DDIMScheduler()
    vqvae = self.dummy_vq_model
    unet = unet.half()
    vqvae = vqvae.half()
    ldm = LDMSuperResolutionPipeline(unet=unet, vqvae=vqvae, scheduler=
        scheduler)
    ldm.to(torch_device)
    ldm.set_progress_bar_config(disable=None)
    init_image = self.dummy_image.to(torch_device)
    image = ldm(init_image, num_inference_steps=2, output_type='np').images
    assert image.shape == (1, 64, 64, 3)
