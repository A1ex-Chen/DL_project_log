def _build_component(self) ->Backbone.Component:
    nasnet_a_large = pretrainedmodels.__dict__['nasnetalarge'](num_classes=
        1000, pretrained='imagenet' if self.pretrained else None)
    conv1 = self.Conv1(nasnet_a_large)
    conv2 = self.Conv2(nasnet_a_large)
    conv3 = self.Conv3(nasnet_a_large, conv1)
    conv4 = self.Conv4(nasnet_a_large, conv3)
    conv5 = self.Conv5(nasnet_a_large)
    num_conv1_out = 96
    num_conv2_out = 168
    num_conv3_out = 1008
    num_conv4_out = 2016
    num_conv5_out = 4032
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
