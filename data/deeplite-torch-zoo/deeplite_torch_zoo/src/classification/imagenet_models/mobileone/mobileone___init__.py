def __init__(self, num_blocks_per_stage: List[int]=[2, 8, 10, 1],
    num_classes: int=1000, width_multipliers: Optional[List[float]]=None,
    inference_mode: bool=False, use_se: bool=False, num_conv_branches: int=1
    ) ->None:
    """Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
    super().__init__()
    assert len(width_multipliers) == 4
    self.inference_mode = inference_mode
    self.in_planes = min(64, int(64 * width_multipliers[0]))
    self.use_se = use_se
    self.num_conv_branches = num_conv_branches
    self.stage0 = MobileOneBlock(c1=3, c2=self.in_planes, k=3, s=2, p=1,
        inference_mode=self.inference_mode)
    self.cur_layer_idx = 1
    self.stage1 = self._make_stage(int(64 * width_multipliers[0]),
        num_blocks_per_stage[0], num_se_blocks=0)
    self.stage2 = self._make_stage(int(128 * width_multipliers[1]),
        num_blocks_per_stage[1], num_se_blocks=0)
    self.stage3 = self._make_stage(int(256 * width_multipliers[2]),
        num_blocks_per_stage[2], num_se_blocks=int(num_blocks_per_stage[2] //
        2) if use_se else 0)
    self.stage4 = self._make_stage(int(512 * width_multipliers[3]),
        num_blocks_per_stage[3], num_se_blocks=num_blocks_per_stage[3] if
        use_se else 0)
    self.gap = nn.AdaptiveAvgPool2d(output_size=1)
    self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)
