@staticmethod
def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs
    ):
    """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
    blocks = []
    for i in range(num_blocks):
        curr_kwargs = {}
        for k, v in kwargs.items():
            if k.endswith('_per_block'):
                assert len(v
                    ) == num_blocks, f"Argument '{k}' of make_stage should have the same length as num_blocks={num_blocks}."
                newk = k[:-len('_per_block')]
                assert newk not in kwargs, f'Cannot call make_stage with both {k} and {newk}!'
                curr_kwargs[newk] = v[i]
            else:
                curr_kwargs[k] = v
        blocks.append(block_class(in_channels=in_channels, out_channels=
            out_channels, **curr_kwargs))
        in_channels = out_channels
    return blocks
