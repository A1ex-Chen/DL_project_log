def _load_ip_adapter_loras(self, state_dicts):
    lora_dicts = {}
    for key_id, name in enumerate(self.attn_processors.keys()):
        for i, state_dict in enumerate(state_dicts):
            if f'{key_id}.to_k_lora.down.weight' in state_dict['ip_adapter']:
                if i not in lora_dicts:
                    lora_dicts[i] = {}
                lora_dicts[i].update({f'unet.{name}.to_k_lora.down.weight':
                    state_dict['ip_adapter'][
                    f'{key_id}.to_k_lora.down.weight']})
                lora_dicts[i].update({f'unet.{name}.to_q_lora.down.weight':
                    state_dict['ip_adapter'][
                    f'{key_id}.to_q_lora.down.weight']})
                lora_dicts[i].update({f'unet.{name}.to_v_lora.down.weight':
                    state_dict['ip_adapter'][
                    f'{key_id}.to_v_lora.down.weight']})
                lora_dicts[i].update({
                    f'unet.{name}.to_out_lora.down.weight': state_dict[
                    'ip_adapter'][f'{key_id}.to_out_lora.down.weight']})
                lora_dicts[i].update({f'unet.{name}.to_k_lora.up.weight':
                    state_dict['ip_adapter'][f'{key_id}.to_k_lora.up.weight']})
                lora_dicts[i].update({f'unet.{name}.to_q_lora.up.weight':
                    state_dict['ip_adapter'][f'{key_id}.to_q_lora.up.weight']})
                lora_dicts[i].update({f'unet.{name}.to_v_lora.up.weight':
                    state_dict['ip_adapter'][f'{key_id}.to_v_lora.up.weight']})
                lora_dicts[i].update({f'unet.{name}.to_out_lora.up.weight':
                    state_dict['ip_adapter'][
                    f'{key_id}.to_out_lora.up.weight']})
    return lora_dicts
