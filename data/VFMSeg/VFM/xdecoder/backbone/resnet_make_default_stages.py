@staticmethod
def make_default_stages(depth, block_class=None, **kwargs):
    """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
    num_blocks_per_stage = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [
        3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}[depth]
    if block_class is None:
        block_class = BasicBlock if depth < 50 else BottleneckBlock
    if depth < 50:
        in_channels = [64, 64, 128, 256]
        out_channels = [64, 128, 256, 512]
    else:
        in_channels = [64, 256, 512, 1024]
        out_channels = [256, 512, 1024, 2048]
    ret = []
    for n, s, i, o in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels,
        out_channels):
        if depth >= 50:
            kwargs['bottleneck_channels'] = o // 4
        ret.append(ResNet.make_stage(block_class=block_class, num_blocks=n,
            stride_per_block=[s] + [1] * (n - 1), in_channels=i,
            out_channels=o, **kwargs))
    return ret
