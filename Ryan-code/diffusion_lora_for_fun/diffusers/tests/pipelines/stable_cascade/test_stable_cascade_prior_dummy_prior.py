@property
def dummy_prior(self):
    torch.manual_seed(0)
    model_kwargs = {'conditioning_dim': 128, 'block_out_channels': (128, 
        128), 'num_attention_heads': (2, 2), 'down_num_layers_per_block': (
        1, 1), 'up_num_layers_per_block': (1, 1), 'switch_level': (False,),
        'clip_image_in_channels': 768, 'clip_text_in_channels': self.
        text_embedder_hidden_size, 'clip_text_pooled_in_channels': self.
        text_embedder_hidden_size, 'dropout': (0.1, 0.1)}
    model = StableCascadeUNet(**model_kwargs)
    return model.eval()
