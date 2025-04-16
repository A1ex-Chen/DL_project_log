def __init__(self, backbone: Backbone, num_body_out: int):
    super().__init__()
    self.conv1 = backbone.component.conv1
    self.conv2 = backbone.component.conv2
    self.conv3 = backbone.component.conv3
    self.conv4 = backbone.component.conv4
    self.conv5 = backbone.component.conv5
    self.fpn = FeaturePyramidNetwork(in_channels_list=[backbone.component.
        num_conv2_out, backbone.component.num_conv3_out, backbone.component
        .num_conv4_out, backbone.component.num_conv5_out], out_channels=
        num_body_out, extra_blocks=LastLevelMaxPool())
