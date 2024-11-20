def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=(
    96, 192, 384, 768), depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
    window_sizes=(7, 7, 14, 7), mlp_ratio=4.0, drop_rate=0.0,
    drop_path_rate=0.1, use_checkpoint=False, mbconv_expand_ratio=4.0,
    local_conv_size=3, layer_lr_decay=1.0):
    """
        Initializes the TinyViT model.

        Args:
            img_size (int, optional): The input image size. Defaults to 224.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of classification classes. Defaults to 1000.
            embed_dims (List[int], optional): List of embedding dimensions per layer. Defaults to [96, 192, 384, 768].
            depths (List[int], optional): List of depths for each layer. Defaults to [2, 2, 6, 2].
            num_heads (List[int], optional): List of number of attention heads per layer. Defaults to [3, 6, 12, 24].
            window_sizes (List[int], optional): List of window sizes for each layer. Defaults to [7, 7, 14, 7].
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Drop path rate for stochastic depth. Defaults to 0.1.
            use_checkpoint (bool, optional): Whether to use checkpointing for efficient memory usage. Defaults to False.
            mbconv_expand_ratio (float, optional): Expansion ratio for MBConv layer. Defaults to 4.0.
            local_conv_size (int, optional): Local convolution kernel size. Defaults to 3.
            layer_lr_decay (float, optional): Layer-wise learning rate decay. Defaults to 1.0.
        """
    super().__init__()
    self.img_size = img_size
    self.num_classes = num_classes
    self.depths = depths
    self.num_layers = len(depths)
    self.mlp_ratio = mlp_ratio
    activation = nn.GELU
    self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0
        ], resolution=img_size, activation=activation)
    patches_resolution = self.patch_embed.patches_resolution
    self.patches_resolution = patches_resolution
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    self.layers = nn.ModuleList()
    for i_layer in range(self.num_layers):
        kwargs = dict(dim=embed_dims[i_layer], input_resolution=(
            patches_resolution[0] // 2 ** (i_layer - 1 if i_layer == 3 else
            i_layer), patches_resolution[1] // 2 ** (i_layer - 1 if i_layer ==
            3 else i_layer)), depth=depths[i_layer], drop_path=dpr[sum(
            depths[:i_layer]):sum(depths[:i_layer + 1])], downsample=
            PatchMerging if i_layer < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint, out_dim=embed_dims[min(i_layer +
            1, len(embed_dims) - 1)], activation=activation)
        if i_layer == 0:
            layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
        else:
            layer = BasicLayer(num_heads=num_heads[i_layer], window_size=
                window_sizes[i_layer], mlp_ratio=self.mlp_ratio, drop=
                drop_rate, local_conv_size=local_conv_size, **kwargs)
        self.layers.append(layer)
    self.norm_head = nn.LayerNorm(embed_dims[-1])
    self.head = nn.Linear(embed_dims[-1], num_classes
        ) if num_classes > 0 else torch.nn.Identity()
    self.apply(self._init_weights)
    self.set_layer_lr_decay(layer_lr_decay)
    self.neck = nn.Sequential(nn.Conv2d(embed_dims[-1], 256, kernel_size=1,
        bias=False), LayerNorm2d(256), nn.Conv2d(256, 256, kernel_size=3,
        padding=1, bias=False), LayerNorm2d(256))
