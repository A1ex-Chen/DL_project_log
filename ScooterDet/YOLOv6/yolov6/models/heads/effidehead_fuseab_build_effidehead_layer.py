def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max
    =16, num_layers=3):
    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]
    head_layers = nn.Sequential(ConvBNSiLU(in_channels=channels_list[chx[0]
        ], out_channels=channels_list[chx[0]], kernel_size=1, stride=1),
        ConvBNSiLU(in_channels=channels_list[chx[0]], out_channels=
        channels_list[chx[0]], kernel_size=3, stride=1), ConvBNSiLU(
        in_channels=channels_list[chx[0]], out_channels=channels_list[chx[0
        ]], kernel_size=3, stride=1), nn.Conv2d(in_channels=channels_list[
        chx[0]], out_channels=num_classes, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[chx[0]], out_channels=4 * (reg_max + 1),
        kernel_size=1), nn.Conv2d(in_channels=channels_list[chx[0]],
        out_channels=num_classes * num_anchors, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[chx[0]], out_channels=4 * num_anchors,
        kernel_size=1), ConvBNSiLU(in_channels=channels_list[chx[1]],
        out_channels=channels_list[chx[1]], kernel_size=1, stride=1),
        ConvBNSiLU(in_channels=channels_list[chx[1]], out_channels=
        channels_list[chx[1]], kernel_size=3, stride=1), ConvBNSiLU(
        in_channels=channels_list[chx[1]], out_channels=channels_list[chx[1
        ]], kernel_size=3, stride=1), nn.Conv2d(in_channels=channels_list[
        chx[1]], out_channels=num_classes, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[chx[1]], out_channels=4 * (reg_max + 1),
        kernel_size=1), nn.Conv2d(in_channels=channels_list[chx[1]],
        out_channels=num_classes * num_anchors, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[chx[1]], out_channels=4 * num_anchors,
        kernel_size=1), ConvBNSiLU(in_channels=channels_list[chx[2]],
        out_channels=channels_list[chx[2]], kernel_size=1, stride=1),
        ConvBNSiLU(in_channels=channels_list[chx[2]], out_channels=
        channels_list[chx[2]], kernel_size=3, stride=1), ConvBNSiLU(
        in_channels=channels_list[chx[2]], out_channels=channels_list[chx[2
        ]], kernel_size=3, stride=1), nn.Conv2d(in_channels=channels_list[
        chx[2]], out_channels=num_classes, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[chx[2]], out_channels=4 * (reg_max + 1),
        kernel_size=1), nn.Conv2d(in_channels=channels_list[chx[2]],
        out_channels=num_classes * num_anchors, kernel_size=1), nn.Conv2d(
        in_channels=channels_list[chx[2]], out_channels=4 * num_anchors,
        kernel_size=1))
    return head_layers
