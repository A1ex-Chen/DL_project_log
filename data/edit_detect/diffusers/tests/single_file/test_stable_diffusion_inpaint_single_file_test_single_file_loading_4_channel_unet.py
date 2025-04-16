def test_single_file_loading_4_channel_unet(self):
    ckpt_path = (
        'https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors'
        )
    pipe = self.pipeline_class.from_single_file(ckpt_path)
    assert pipe.unet.config.in_channels == 4
