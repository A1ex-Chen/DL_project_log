@property
def dummy_vae(self):
    torch.manual_seed(0)
    model = AutoencoderKL(block_out_channels=[32, 32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D', 'DownEncoderBlock2D'], up_block_types=[
        'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        latent_channels=4)
    return model
