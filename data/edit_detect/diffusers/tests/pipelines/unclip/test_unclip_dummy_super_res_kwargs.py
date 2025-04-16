@property
def dummy_super_res_kwargs(self):
    return {'sample_size': 64, 'layers_per_block': 1, 'down_block_types': (
        'ResnetDownsampleBlock2D', 'ResnetDownsampleBlock2D'),
        'up_block_types': ('ResnetUpsampleBlock2D', 'ResnetUpsampleBlock2D'
        ), 'block_out_channels': (self.block_out_channels_0, self.
        block_out_channels_0 * 2), 'in_channels': 6, 'out_channels': 3}
