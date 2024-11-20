@property
def dummy_prior(self):
    torch.manual_seed(0)
    model_kwargs = {'conditioning_dim': 128, 'block_out_channels': (128, 
        128), 'num_attention_heads': (2, 2), 'down_num_layers_per_block': (
        1, 1), 'up_num_layers_per_block': (1, 1), 'clip_image_in_channels':
        768, 'switch_level': (False,), 'clip_text_in_channels': self.
        text_embedder_hidden_size, 'clip_text_pooled_in_channels': self.
        text_embedder_hidden_size}
    model = StableCascadeUNet(**model_kwargs)
    return model.eval()
