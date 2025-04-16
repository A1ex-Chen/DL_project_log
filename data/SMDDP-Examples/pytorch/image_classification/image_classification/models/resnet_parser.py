def parser(self, name):
    p = super().parser(name)
    p.add_argument('--num_classes', metavar='N', default=self.num_classes,
        type=int, help='number of classes')
    p.add_argument('--last_bn_0_init', metavar='True|False', default=self.
        last_bn_0_init, type=bool)
    p.add_argument('--conv_init', default=self.conv_init, choices=['fan_in',
        'fan_out'], type=str, help=
        'initialization mode for convolutional layers, see https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_'
        )
    p.add_argument('--trt', metavar='True|False', default=self.trt, type=bool)
    return p
