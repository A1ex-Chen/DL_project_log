def __init__(self, c1, c2, s=1, act='relu'):
    super(SqnxtUnit, self).__init__()
    if s == 2:
        reduction_den = 1
        self.resize_identity = True
    elif c1 > c2:
        reduction_den = 4
        self.resize_identity = True
    elif c1 <= c2:
        reduction_den = 2
        self.resize_identity = False
    self.conv1 = ConvBnAct(c1=c1, c2=c1 // reduction_den, k=1, s=s, b=True,
        act=act)
    self.conv2 = ConvBnAct(c1=c1 // reduction_den, c2=c1 // (2 *
        reduction_den), k=1, b=True, act=act)
    self.conv3 = ConvBnAct(c1=c1 // (2 * reduction_den), c2=c1 //
        reduction_den, k=(1, 3), s=1, p=(0, 1), act=act)
    self.conv4 = ConvBnAct(c1=c1 // reduction_den, c2=c1 // reduction_den,
        k=(3, 1), s=1, p=(1, 0), act=act)
    self.conv5 = ConvBnAct(c1=c1 // reduction_den, c2=c2, k=1, b=True, act=act)
    if self.resize_identity:
        self.identity_conv = ConvBnAct(c1=c1, c2=c2, s=s, act=act)
    self.activ = get_activation(act)
