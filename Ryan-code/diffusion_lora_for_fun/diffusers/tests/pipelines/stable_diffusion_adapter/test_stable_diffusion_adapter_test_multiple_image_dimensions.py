@parameterized.expand([((4 * 8 + 1) * 8,), ((4 * 4 + 1) * 16,), ((4 * 2 + 1
    ) * 32,), ((4 * 1 + 1) * 64,)])
def test_multiple_image_dimensions(self, dim):
    """Test that the T2I-Adapter pipeline supports any input dimension that
        is divisible by the adapter's `downscale_factor`. This test was added in
        response to an issue where the T2I Adapter's downscaling padding
        behavior did not match the UNet's behavior.

        Note that we have selected `dim` values to produce odd resolutions at
        each downscaling level.
        """
    components = self.get_dummy_components_with_full_downscaling()
    sd_pipe = StableDiffusionAdapterPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device, height=dim, width=dim)
    image = sd_pipe(**inputs).images
    assert image.shape == (1, dim, dim, 3)
