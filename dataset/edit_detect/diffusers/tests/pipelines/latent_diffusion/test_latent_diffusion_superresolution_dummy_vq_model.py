@property
def dummy_vq_model(self):
    torch.manual_seed(0)
    model = VQModel(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=3)
    return model
