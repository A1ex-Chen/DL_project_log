def _convert_ip_adapter_attn_to_diffusers(self, state_dicts,
    low_cpu_mem_usage=False):
    from ..models.attention_processor import AttnProcessor, AttnProcessor2_0, IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
    if low_cpu_mem_usage:
        if is_accelerate_available():
            from accelerate import init_empty_weights
        else:
            low_cpu_mem_usage = False
            logger.warning(
                """Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with: 
```
pip install accelerate
```
."""
                )
    if low_cpu_mem_usage is True and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set `low_cpu_mem_usage=False`.'
            )
    attn_procs = {}
    key_id = 1
    init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
    for name in self.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else self.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = self.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(self.config.block_out_channels))[
                block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = self.config.block_out_channels[block_id]
        if cross_attention_dim is None or 'motion_modules' in name:
            attn_processor_class = AttnProcessor2_0 if hasattr(F,
                'scaled_dot_product_attention') else AttnProcessor
            attn_procs[name] = attn_processor_class()
        else:
            attn_processor_class = IPAdapterAttnProcessor2_0 if hasattr(F,
                'scaled_dot_product_attention') else IPAdapterAttnProcessor
            num_image_text_embeds = []
            for state_dict in state_dicts:
                if 'proj.weight' in state_dict['image_proj']:
                    num_image_text_embeds += [4]
                elif 'proj.3.weight' in state_dict['image_proj']:
                    num_image_text_embeds += [257]
                elif 'perceiver_resampler.proj_in.weight' in state_dict[
                    'image_proj']:
                    num_image_text_embeds += [4]
                elif 'norm.weight' in state_dict['image_proj']:
                    num_image_text_embeds += [4]
                else:
                    num_image_text_embeds += [state_dict['image_proj'][
                        'latents'].shape[1]]
            with init_context():
                attn_procs[name] = attn_processor_class(hidden_size=
                    hidden_size, cross_attention_dim=cross_attention_dim,
                    scale=1.0, num_tokens=num_image_text_embeds)
            value_dict = {}
            for i, state_dict in enumerate(state_dicts):
                value_dict.update({f'to_k_ip.{i}.weight': state_dict[
                    'ip_adapter'][f'{key_id}.to_k_ip.weight']})
                value_dict.update({f'to_v_ip.{i}.weight': state_dict[
                    'ip_adapter'][f'{key_id}.to_v_ip.weight']})
            if not low_cpu_mem_usage:
                attn_procs[name].load_state_dict(value_dict)
            else:
                device = next(iter(value_dict.values())).device
                dtype = next(iter(value_dict.values())).dtype
                load_model_dict_into_meta(attn_procs[name], value_dict,
                    device=device, dtype=dtype)
            key_id += 2
    return attn_procs
