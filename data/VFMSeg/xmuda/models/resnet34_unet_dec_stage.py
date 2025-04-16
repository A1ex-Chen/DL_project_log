@staticmethod
def dec_stage(enc_stage, num_concat):
    in_channels = enc_stage[0].conv1.in_channels
    out_channels = enc_stage[-1].conv2.out_channels
    conv = nn.Sequential(nn.Conv2d(num_concat * out_channels, out_channels,
        kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(
        inplace=True))
    t_conv = nn.Sequential(nn.ConvTranspose2d(out_channels, in_channels,
        kernel_size=2, stride=2), nn.BatchNorm2d(in_channels), nn.ReLU(
        inplace=True))
    return conv, t_conv
