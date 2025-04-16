def _build_component(self) ->Backbone.Component:
    efficientnet_v2_s = torchvision.models.efficientnet_v2_s(pretrained=
        self.pretrained)
    conv1 = efficientnet_v2_s.features[:2]
    conv2 = efficientnet_v2_s.features[2:3]
    conv3 = efficientnet_v2_s.features[3:4]
    conv4 = efficientnet_v2_s.features[4:6]
    conv5 = efficientnet_v2_s.features[6:]
    num_conv1_out = 48
    num_conv2_out = 64
    num_conv3_out = 128
    num_conv4_out = 256
    num_conv5_out = 1280
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
