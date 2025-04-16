@register_to_config
def __init__(self, *, param_names: Tuple[str]=('nerstf.mlp.0.weight',
    'nerstf.mlp.1.weight', 'nerstf.mlp.2.weight', 'nerstf.mlp.3.weight'),
    param_shapes: Tuple[Tuple[int]]=((256, 93), (256, 256), (256, 256), (
    256, 256)), d_latent: int=1024, d_hidden: int=256, n_output: int=12,
    n_hidden_layers: int=6, act_fn: str='swish', insert_direction_at: int=4,
    background: Tuple[float]=(255.0, 255.0, 255.0)):
    super().__init__()
    self.params_proj = ShapEParamsProjModel(param_names=param_names,
        param_shapes=param_shapes, d_latent=d_latent)
    self.mlp = MLPNeRSTFModel(d_hidden, n_output, n_hidden_layers, act_fn,
        insert_direction_at)
    self.void = VoidNeRFModel(background=background, channel_scale=255.0)
    self.volume = BoundingBoxVolume(bbox_max=[1.0, 1.0, 1.0], bbox_min=[-
        1.0, -1.0, -1.0])
    self.mesh_decoder = MeshDecoder()
