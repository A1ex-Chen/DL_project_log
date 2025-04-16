def convert_asymmetric_autoencoder_kl_state_dict(original_state_dict: Dict[
    str, Any]) ->Dict[str, Any]:
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if k.startswith('encoder.'):
            converted_state_dict[k.replace('encoder.down.',
                'encoder.down_blocks.').replace('encoder.mid.',
                'encoder.mid_block.').replace('encoder.norm_out.',
                'encoder.conv_norm_out.').replace('.downsample.',
                '.downsamplers.0.').replace('.nin_shortcut.',
                '.conv_shortcut.').replace('.block.', '.resnets.').replace(
                '.block_1.', '.resnets.0.').replace('.block_2.',
                '.resnets.1.').replace('.attn_1.k.', '.attentions.0.to_k.')
                .replace('.attn_1.q.', '.attentions.0.to_q.').replace(
                '.attn_1.v.', '.attentions.0.to_v.').replace(
                '.attn_1.proj_out.', '.attentions.0.to_out.0.').replace(
                '.attn_1.norm.', '.attentions.0.group_norm.')] = v
        elif k.startswith('decoder.') and 'up_layers' not in k:
            converted_state_dict[k.replace('decoder.encoder.',
                'decoder.condition_encoder.').replace('.norm_out.',
                '.conv_norm_out.').replace('.up.0.', '.up_blocks.3.').
                replace('.up.1.', '.up_blocks.2.').replace('.up.2.',
                '.up_blocks.1.').replace('.up.3.', '.up_blocks.0.').replace
                ('.block.', '.resnets.').replace('mid', 'mid_block').
                replace('.0.upsample.', '.0.upsamplers.0.').replace(
                '.1.upsample.', '.1.upsamplers.0.').replace('.2.upsample.',
                '.2.upsamplers.0.').replace('.nin_shortcut.',
                '.conv_shortcut.').replace('.block_1.', '.resnets.0.').
                replace('.block_2.', '.resnets.1.').replace('.attn_1.k.',
                '.attentions.0.to_k.').replace('.attn_1.q.',
                '.attentions.0.to_q.').replace('.attn_1.v.',
                '.attentions.0.to_v.').replace('.attn_1.proj_out.',
                '.attentions.0.to_out.0.').replace('.attn_1.norm.',
                '.attentions.0.group_norm.')] = v
        elif k.startswith('quant_conv.'):
            converted_state_dict[k] = v
        elif k.startswith('post_quant_conv.'):
            converted_state_dict[k] = v
        else:
            print(f'  skipping key `{k}`')
    for k, v in converted_state_dict.items():
        if (k.startswith('encoder.mid_block.attentions.0') or k.startswith(
            'decoder.mid_block.attentions.0')) and k.endswith('weight') and (
            'to_q' in k or 'to_k' in k or 'to_v' in k or 'to_out' in k):
            converted_state_dict[k] = converted_state_dict[k][:, :, 0, 0]
    return converted_state_dict
