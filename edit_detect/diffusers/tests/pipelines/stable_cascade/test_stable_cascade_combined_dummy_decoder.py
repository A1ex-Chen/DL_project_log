@property
def dummy_decoder(self):
    torch.manual_seed(0)
    model_kwargs = {'in_channels': 4, 'out_channels': 4, 'conditioning_dim':
        128, 'block_out_channels': (16, 32, 64, 128), 'num_attention_heads':
        (-1, -1, 1, 2), 'down_num_layers_per_block': (1, 1, 1, 1),
        'up_num_layers_per_block': (1, 1, 1, 1),
        'down_blocks_repeat_mappers': (1, 1, 1, 1),
        'up_blocks_repeat_mappers': (3, 3, 2, 2), 'block_types_per_layer':
        (('SDCascadeResBlock', 'SDCascadeTimestepBlock'), (
        'SDCascadeResBlock', 'SDCascadeTimestepBlock'), (
        'SDCascadeResBlock', 'SDCascadeTimestepBlock', 'SDCascadeAttnBlock'
        ), ('SDCascadeResBlock', 'SDCascadeTimestepBlock',
        'SDCascadeAttnBlock')), 'switch_level': None,
        'clip_text_pooled_in_channels': 32, 'dropout': (0.1, 0.1, 0.1, 0.1)}
    model = StableCascadeUNet(**model_kwargs)
    return model.eval()
