def __init__(self, c1: int, c2: int, k: int=3, s: int=1, p: int=1, d: int=1,
    g: int=1, act='relu', inference_mode: bool=False, use_se: bool=False,
    num_conv_branches: int=1) ->None:
    """Construct a MobileOneBlock module.
        Mobile block builds on the MobileNet-V1 [1] block of 3x3 depthwise convolution followed
        by 1x1 pointwise convolutions
        :param c1: Number of channels in the input.
        :param c2: Number of channels produced by the block.
        :param k: Size of the convolution kernel.
        :param s: Stride size.
        :param p: Zero-padding size.
        :param d: Kernel dilation factor.
        :param g: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
    super().__init__()
    self.depthwise_conv = MobileOneBlock(c1=c1, c2=c1, k=k, s=s, p=p, d=d,
        g=c1, act=act, inference_mode=inference_mode, use_se=use_se,
        num_conv_branches=num_conv_branches)
    self.pointwise_conv = MobileOneBlock(c1=c1, c2=c2, k=1, s=s, p=0, d=d,
        g=g, act=act, inference_mode=inference_mode, use_se=use_se,
        num_conv_branches=num_conv_branches)
