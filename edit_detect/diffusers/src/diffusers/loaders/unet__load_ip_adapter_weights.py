def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
    if not isinstance(state_dicts, list):
        state_dicts = [state_dicts]
    self.encoder_hid_proj = None
    attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts,
        low_cpu_mem_usage=low_cpu_mem_usage)
    self.set_attn_processor(attn_procs)
    image_projection_layers = []
    for state_dict in state_dicts:
        image_projection_layer = (self.
            _convert_ip_adapter_image_proj_to_diffusers(state_dict[
            'image_proj'], low_cpu_mem_usage=low_cpu_mem_usage))
        image_projection_layers.append(image_projection_layer)
    self.encoder_hid_proj = MultiIPAdapterImageProjection(
        image_projection_layers)
    self.config.encoder_hid_dim_type = 'ip_image_proj'
    self.to(dtype=self.dtype, device=self.device)
