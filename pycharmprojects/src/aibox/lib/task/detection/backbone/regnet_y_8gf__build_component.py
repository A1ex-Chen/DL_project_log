def _build_component(self) ->Backbone.Component:
    regnet_y_8gf = torchvision.models.regnet_y_8gf(pretrained=self.pretrained)
    conv1 = regnet_y_8gf.stem
    conv2 = regnet_y_8gf.trunk_output[:1]
    conv3 = regnet_y_8gf.trunk_output[1:2]
    conv4 = regnet_y_8gf.trunk_output[2:3]
    conv5 = regnet_y_8gf.trunk_output[3:]
    num_conv1_out = 32
    num_conv2_out = 224
    num_conv3_out = 448
    num_conv4_out = 896
    num_conv5_out = 2016
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
