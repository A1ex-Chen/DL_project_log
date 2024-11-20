def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16
    ):
    head_layers = nn.Sequential(ConvBNSiLU(in_channels=channels_list[6],
        out_channels=channels_list[6], kernel_size=1, stride=1), ConvBNSiLU
        (in_channels=channels_list[6], out_channels=channels_list[6],
        kernel_size=3, stride=1), ConvBNSiLU(in_channels=channels_list[6],
        out_channels=channels_list[6], kernel_size=3, stride=1), nn.Conv2d(
        in_channels=channels_list[6], out_channels=num_classes *
        num_anchors, kernel_size=1), nn.Conv2d(in_channels=channels_list[6],
        out_channels=4 * (reg_max + num_anchors), kernel_size=1), nn.Conv2d
        (in_channels=channels_list[6], out_channels=4 * num_anchors,
        kernel_size=1), ConvBNSiLU(in_channels=channels_list[8],
        out_channels=channels_list[8], kernel_size=1, stride=1), ConvBNSiLU
        (in_channels=channels_list[8], out_channels=channels_list[8],
        kernel_size=3, stride=1), ConvBNSiLU(in_channels=channels_list[8],
        out_channels=channels_list[8], kernel_size=3, stride=1), nn.Conv2d(
        in_channels=channels_list[8], out_channels=num_classes *
        num_anchors, kernel_size=1), nn.Conv2d(in_channels=channels_list[8],
        out_channels=4 * (reg_max + num_anchors), kernel_size=1), nn.Conv2d
        (in_channels=channels_list[8], out_channels=4 * num_anchors,
        kernel_size=1), ConvBNSiLU(in_channels=channels_list[10],
        out_channels=channels_list[10], kernel_size=1, stride=1),
        ConvBNSiLU(in_channels=channels_list[10], out_channels=
        channels_list[10], kernel_size=3, stride=1), ConvBNSiLU(in_channels
        =channels_list[10], out_channels=channels_list[10], kernel_size=3,
        stride=1), nn.Conv2d(in_channels=channels_list[10], out_channels=
        num_classes * num_anchors, kernel_size=1), nn.Conv2d(in_channels=
        channels_list[10], out_channels=4 * (reg_max + num_anchors),
        kernel_size=1), nn.Conv2d(in_channels=channels_list[10],
        out_channels=4 * num_anchors, kernel_size=1))
    return head_layers
