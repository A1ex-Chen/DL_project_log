def __init__(self, pretrained=True):
    super(UNetResNet34, self).__init__()
    net = resnet34(pretrained)
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=
        False)
    self.conv1.weight.data = net.conv1.weight.data
    self.bn1 = net.bn1
    self.relu = net.relu
    self.maxpool = net.maxpool
    self.layer1 = net.layer1
    self.layer2 = net.layer2
    self.layer3 = net.layer3
    self.layer4 = net.layer4
    _, self.dec_t_conv_stage5 = self.dec_stage(self.layer4, num_concat=1)
    self.dec_conv_stage4, self.dec_t_conv_stage4 = self.dec_stage(self.
        layer3, num_concat=2)
    self.dec_conv_stage3, self.dec_t_conv_stage3 = self.dec_stage(self.
        layer2, num_concat=2)
    self.dec_conv_stage2, self.dec_t_conv_stage2 = self.dec_stage(self.
        layer1, num_concat=2)
    self.dec_conv_stage1 = nn.Conv2d(2 * 64, 64, kernel_size=3, padding=1)
    self.dropout = nn.Dropout(p=0.4)
