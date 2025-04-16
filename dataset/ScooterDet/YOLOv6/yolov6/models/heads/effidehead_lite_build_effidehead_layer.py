def build_effidehead_layer(channels_list, num_anchors, num_classes, num_layers
    ):
    head_layers = nn.Sequential(DPBlock(in_channel=channels_list[0],
        out_channel=channels_list[0], kernel_size=5, stride=1), DPBlock(
        in_channel=channels_list[0], out_channel=channels_list[0],
        kernel_size=5, stride=1), DPBlock(in_channel=channels_list[0],
        out_channel=channels_list[0], kernel_size=5, stride=1), nn.Conv2d(
        in_channels=channels_list[0], out_channels=num_classes *
        num_anchors, kernel_size=1), nn.Conv2d(in_channels=channels_list[0],
        out_channels=4 * num_anchors, kernel_size=1), DPBlock(in_channel=
        channels_list[1], out_channel=channels_list[1], kernel_size=5,
        stride=1), DPBlock(in_channel=channels_list[1], out_channel=
        channels_list[1], kernel_size=5, stride=1), DPBlock(in_channel=
        channels_list[1], out_channel=channels_list[1], kernel_size=5,
        stride=1), nn.Conv2d(in_channels=channels_list[1], out_channels=
        num_classes * num_anchors, kernel_size=1), nn.Conv2d(in_channels=
        channels_list[1], out_channels=4 * num_anchors, kernel_size=1),
        DPBlock(in_channel=channels_list[2], out_channel=channels_list[2],
        kernel_size=5, stride=1), DPBlock(in_channel=channels_list[2],
        out_channel=channels_list[2], kernel_size=5, stride=1), DPBlock(
        in_channel=channels_list[2], out_channel=channels_list[2],
        kernel_size=5, stride=1), nn.Conv2d(in_channels=channels_list[2],
        out_channels=num_classes * num_anchors, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[2], out_channels=4 * num_anchors,
        kernel_size=1))
    if num_layers == 4:
        head_layers.add_module('stem3', DPBlock(in_channel=channels_list[3],
            out_channel=channels_list[3], kernel_size=5, stride=1))
        head_layers.add_module('cls_conv3', DPBlock(in_channel=
            channels_list[3], out_channel=channels_list[3], kernel_size=5,
            stride=1))
        head_layers.add_module('reg_conv3', DPBlock(in_channel=
            channels_list[3], out_channel=channels_list[3], kernel_size=5,
            stride=1))
        head_layers.add_module('cls_pred3', nn.Conv2d(in_channels=
            channels_list[3], out_channels=num_classes * num_anchors,
            kernel_size=1))
        head_layers.add_module('reg_pred3', nn.Conv2d(in_channels=
            channels_list[3], out_channels=4 * num_anchors, kernel_size=1))
    return head_layers
