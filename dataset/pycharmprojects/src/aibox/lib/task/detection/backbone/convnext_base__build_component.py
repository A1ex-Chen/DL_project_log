def _build_component(self) ->Backbone.Component:
    convnext_base = torchvision.models.convnext_base(pretrained=self.pretrained
        )
    conv1 = convnext_base.features[:1]
    conv2 = convnext_base.features[1:2]
    conv3 = convnext_base.features[2:4]
    conv4 = convnext_base.features[4:6]
    conv5 = convnext_base.features[6:]
    num_conv1_out = 128
    num_conv2_out = 128
    num_conv3_out = 256
    num_conv4_out = 512
    num_conv5_out = 1024
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
