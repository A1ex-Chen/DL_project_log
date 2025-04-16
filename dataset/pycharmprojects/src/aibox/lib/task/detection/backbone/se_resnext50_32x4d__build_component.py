def _build_component(self) ->Backbone.Component:
    se_resnext50_32x4d = pretrainedmodels.__dict__['se_resnext50_32x4d'](
        num_classes=1000, pretrained='imagenet' if self.pretrained else None)
    children = list(se_resnext50_32x4d.children())
    conv1 = children[0][:-1]
    conv2 = nn.Sequential(children[0][-1], *children[1])
    conv3 = children[2]
    conv4 = children[3]
    conv5 = children[4]
    num_conv1_out = 64
    num_conv2_out = 256
    num_conv3_out = 512
    num_conv4_out = 1024
    num_conv5_out = 2048
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
