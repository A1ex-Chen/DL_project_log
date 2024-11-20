def get_time_embed(self, sample: torch.Tensor, timestep: Union[torch.Tensor,
    float, int]) ->Optional[torch.Tensor]:
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == 'mps'
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device
            )
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    return t_emb
