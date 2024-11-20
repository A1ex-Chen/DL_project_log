def _load_ip_adapter_weights(self, state_dict):
    num_image_text_embeds = 4
    self.unet.encoder_hid_proj = None
    attn_procs = {}
    lora_dict = {}
    key_id = 0
    for name in self.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else self.unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = self.unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(self.unet.config.block_out_channels))[
                block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = self.unet.config.block_out_channels[block_id]
        if cross_attention_dim is None or 'motion_modules' in name:
            attn_processor_class = AttnProcessor2_0 if hasattr(F,
                'scaled_dot_product_attention') else AttnProcessor
            attn_procs[name] = attn_processor_class()
            lora_dict.update({f'unet.{name}.to_k_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_k_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_q_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_q_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_v_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_v_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_out_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_out_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_k_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_k_lora.up.weight']})
            lora_dict.update({f'unet.{name}.to_q_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_q_lora.up.weight']})
            lora_dict.update({f'unet.{name}.to_v_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_v_lora.up.weight']})
            lora_dict.update({f'unet.{name}.to_out_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_out_lora.up.weight']})
            key_id += 1
        else:
            attn_processor_class = IPAdapterAttnProcessor2_0 if hasattr(F,
                'scaled_dot_product_attention') else IPAdapterAttnProcessor
            attn_procs[name] = attn_processor_class(hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim, scale=1.0,
                num_tokens=num_image_text_embeds).to(dtype=self.dtype,
                device=self.device)
            lora_dict.update({f'unet.{name}.to_k_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_k_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_q_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_q_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_v_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_v_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_out_lora.down.weight':
                state_dict['ip_adapter'][f'{key_id}.to_out_lora.down.weight']})
            lora_dict.update({f'unet.{name}.to_k_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_k_lora.up.weight']})
            lora_dict.update({f'unet.{name}.to_q_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_q_lora.up.weight']})
            lora_dict.update({f'unet.{name}.to_v_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_v_lora.up.weight']})
            lora_dict.update({f'unet.{name}.to_out_lora.up.weight':
                state_dict['ip_adapter'][f'{key_id}.to_out_lora.up.weight']})
            value_dict = {}
            value_dict.update({'to_k_ip.0.weight': state_dict['ip_adapter']
                [f'{key_id}.to_k_ip.weight']})
            value_dict.update({'to_v_ip.0.weight': state_dict['ip_adapter']
                [f'{key_id}.to_v_ip.weight']})
            attn_procs[name].load_state_dict(value_dict)
            key_id += 1
    self.unet.set_attn_processor(attn_procs)
    self.load_lora_weights(lora_dict, adapter_name='faceid')
    self.set_adapters(['faceid'], adapter_weights=[1.0])
    image_projection = self.convert_ip_adapter_image_proj_to_diffusers(
        state_dict['image_proj'])
    image_projection_layers = [image_projection.to(device=self.device,
        dtype=self.dtype)]
    self.unet.encoder_hid_proj = MultiIPAdapterImageProjection(
        image_projection_layers)
    self.unet.config.encoder_hid_dim_type = 'ip_image_proj'
