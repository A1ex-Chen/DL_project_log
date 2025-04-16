def _build_component(self) ->Backbone.Component:
    wide_resnet50_2 = torchvision.models.wide_resnet50_2(pretrained=self.
        pretrained)
    children = list(wide_resnet50_2.children())
    conv1 = nn.Sequential(*children[:3])
    conv2 = nn.Sequential(*children[3:5])
    conv3 = children[5]
    conv4 = children[6]
    conv5 = children[7]
    num_conv1_out = 64
    num_conv2_out = 256
    num_conv3_out = 512
    num_conv4_out = 1024
    num_conv5_out = 2048
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
