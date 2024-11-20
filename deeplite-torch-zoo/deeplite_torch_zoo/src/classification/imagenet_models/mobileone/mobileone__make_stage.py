def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int
    ) ->nn.Sequential:
    """Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
    strides = [2] + [1] * (num_blocks - 1)
    blocks = []
    for ix, stride in enumerate(strides):
        use_se = False
        if num_se_blocks > num_blocks:
            raise ValueError(
                'Number of SE blocks cannot exceed number of layers.')
        if ix >= num_blocks - num_se_blocks:
            use_se = True
        blocks.append(MobileOneBlock(c1=self.in_planes, c2=self.in_planes,
            k=3, s=stride, p=1, g=self.in_planes, inference_mode=self.
            inference_mode, use_se=use_se, num_conv_branches=self.
            num_conv_branches))
        blocks.append(MobileOneBlock(c1=self.in_planes, c2=planes, k=1, s=1,
            p=0, g=1, inference_mode=self.inference_mode, use_se=use_se,
            num_conv_branches=self.num_conv_branches))
        self.in_planes = planes
        self.cur_layer_idx += 1
    return nn.Sequential(*blocks)
