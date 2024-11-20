def _build_component(self) ->Backbone.Component:
    efficientnet_b7 = torchvision.models.efficientnet_b7(pretrained=self.
        pretrained)
    conv1 = efficientnet_b7.features[:2]
    conv2 = efficientnet_b7.features[2:3]
    conv3 = efficientnet_b7.features[3:6]
    conv4 = efficientnet_b7.features[6:8]
    conv5 = efficientnet_b7.features[8:]
    num_conv1_out = 32
    num_conv2_out = 48
    num_conv3_out = 224
    num_conv4_out = 640
    num_conv5_out = 2560
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
