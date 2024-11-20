def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
    mlp_func = partial(GenericMLP, norm_fn_name='bn1d', activation='relu',
        use_conv=True, hidden_dims=[decoder_dim, decoder_dim], dropout=
        mlp_dropout, input_dim=decoder_dim)
    semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)
    center_head = mlp_func(output_dim=3)
    size_head = mlp_func(output_dim=3)
    angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
    angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)
    mlp_heads = [('sem_cls_head', semcls_head), ('center_head', center_head
        ), ('size_head', size_head), ('angle_cls_head', angle_cls_head), (
        'angle_residual_head', angle_reg_head)]
    self.mlp_heads = nn.ModuleDict(mlp_heads)
