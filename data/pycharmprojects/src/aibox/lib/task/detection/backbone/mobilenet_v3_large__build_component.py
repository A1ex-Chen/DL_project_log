def _build_component(self) ->Backbone.Component:
    mobilenet_v3_large = torchvision.models.mobilenet_v3_large(pretrained=
        self.pretrained)
    conv1 = mobilenet_v3_large.features[:2]
    conv2 = mobilenet_v3_large.features[2:4]
    conv3 = mobilenet_v3_large.features[4:7]
    conv4 = mobilenet_v3_large.features[7:13]
    conv5 = mobilenet_v3_large.features[13:]
    num_conv1_out = 16
    num_conv2_out = 24
    num_conv3_out = 40
    num_conv4_out = 112
    num_conv5_out = 960
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
