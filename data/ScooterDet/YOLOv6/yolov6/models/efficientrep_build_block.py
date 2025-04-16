@staticmethod
def build_block(num_repeat, in_channels, mid_channels, out_channels):
    block_list = nn.Sequential()
    for i in range(num_repeat):
        if i == 0:
            block = Lite_EffiBlockS2(in_channels=in_channels, mid_channels=
                mid_channels, out_channels=out_channels, stride=2)
        else:
            block = Lite_EffiBlockS1(in_channels=out_channels, mid_channels
                =mid_channels, out_channels=out_channels, stride=1)
        block_list.add_module(str(i), block)
    return block_list
