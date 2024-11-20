def parser(self, name):
    p = super().parser(name)
    p.add_argument('--num_classes', metavar='N', default=self.num_classes,
        type=int, help='number of classes')
    p.add_argument('--conv_init', default=self.conv_init, choices=['fan_in',
        'fan_out'], type=str, help=
        'initialization mode for convolutional layers, see https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_'
        )
    p.add_argument('--bn_momentum', default=self.bn_momentum, type=float,
        help='Batch Norm momentum')
    p.add_argument('--bn_epsilon', default=self.bn_epsilon, type=float,
        help='Batch Norm epsilon')
    p.add_argument('--survival_prob', default=self.survival_prob, type=
        float, help='Survival probability for stochastic depth')
    p.add_argument('--dropout', default=self.dropout, type=float, help=
        'Dropout drop prob')
    p.add_argument('--trt', metavar='True|False', default=self.trt, type=bool)
    return p
