@torch.no_grad()
def decode_to_mesh(self, latents, device, grid_size: int=128,
    query_batch_size: int=4096, texture_channels: Tuple=('R', 'G', 'B')):
    projected_params = self.params_proj(latents)
    for name, param in self.mlp.state_dict().items():
        if f'nerstf.{name}' in projected_params.keys():
            param.copy_(projected_params[f'nerstf.{name}'].squeeze(0))
    query_points = volume_query_points(self.volume, grid_size)
    query_positions = query_points[None].repeat(1, 1, 1).to(device=device,
        dtype=self.mlp.dtype)
    fields = []
    for idx in range(0, query_positions.shape[1], query_batch_size):
        query_batch = query_positions[:, idx:idx + query_batch_size]
        model_out = self.mlp(position=query_batch, direction=None, ts=None,
            nerf_level='fine', rendering_mode='stf')
        fields.append(model_out.signed_distance)
    fields = torch.cat(fields, dim=1)
    fields = fields.float()
    assert len(fields.shape) == 3 and fields.shape[-1
        ] == 1, f'expected [meta_batch x inner_batch] SDF results, but got {fields.shape}'
    fields = fields.reshape(1, *([grid_size] * 3))
    full_grid = torch.zeros(1, grid_size + 2, grid_size + 2, grid_size + 2,
        device=fields.device, dtype=fields.dtype)
    full_grid.fill_(-1.0)
    full_grid[:, 1:-1, 1:-1, 1:-1] = fields
    fields = full_grid
    raw_meshes = []
    mesh_mask = []
    for field in fields:
        raw_mesh = self.mesh_decoder(field, self.volume.bbox_min, self.
            volume.bbox_max - self.volume.bbox_min)
        mesh_mask.append(True)
        raw_meshes.append(raw_mesh)
    mesh_mask = torch.tensor(mesh_mask, device=fields.device)
    max_vertices = max(len(m.verts) for m in raw_meshes)
    texture_query_positions = torch.stack([m.verts[torch.arange(0,
        max_vertices) % len(m.verts)] for m in raw_meshes], dim=0)
    texture_query_positions = texture_query_positions.to(device=device,
        dtype=self.mlp.dtype)
    textures = []
    for idx in range(0, texture_query_positions.shape[1], query_batch_size):
        query_batch = texture_query_positions[:, idx:idx + query_batch_size]
        texture_model_out = self.mlp(position=query_batch, direction=None,
            ts=None, nerf_level='fine', rendering_mode='stf')
        textures.append(texture_model_out.channels)
    textures = torch.cat(textures, dim=1)
    textures = _convert_srgb_to_linear(textures)
    textures = textures.float()
    assert len(textures.shape) == 3 and textures.shape[-1] == len(
        texture_channels
        ), f'expected [meta_batch x inner_batch x texture_channels] field results, but got {textures.shape}'
    for m, texture in zip(raw_meshes, textures):
        texture = texture[:len(m.verts)]
        m.vertex_channels = dict(zip(texture_channels, texture.unbind(-1)))
    return raw_meshes[0]
