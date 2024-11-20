def _build_component(self) ->Backbone.Component:
    pnasnet_5_large = pretrainedmodels.__dict__['pnasnet5large'](num_classes
        =1000, pretrained='imagenet' if self.pretrained else None)
    conv1 = self.Conv1(pnasnet_5_large)
    conv2 = self.Conv2(pnasnet_5_large)
    conv3 = self.Conv3(pnasnet_5_large, conv1)
    conv4 = self.Conv4(pnasnet_5_large, conv3)
    conv5 = self.Conv5(pnasnet_5_large)
    num_conv1_out = 96
    num_conv2_out = 168
    num_conv3_out = 1008
    num_conv4_out = 2160
    num_conv5_out = 4320
    return Backbone.Component(conv1, conv2, conv3, conv4, conv5,
        num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out,
        num_conv5_out)
