def get_unet_lora_target_modules(unet, use_blora, target_blocks=None):
    if use_blora:
        content_b_lora_blocks = 'unet.up_blocks.0.attentions.0'
        style_b_lora_blocks = 'unet.up_blocks.0.attentions.1'
        target_blocks = [content_b_lora_blocks, style_b_lora_blocks]
    try:
        blocks = ['.'.join(blk.split('.')[1:]) for blk in target_blocks]
        attns = [attn_processor_name.rsplit('.', 1)[0] for 
            attn_processor_name, _ in unet.attn_processors.items() if
            is_belong_to_blocks(attn_processor_name, blocks)]
        target_modules = [f'{attn}.{mat}' for mat in ['to_k', 'to_q',
            'to_v', 'to_out.0'] for attn in attns]
        return target_modules
    except Exception as e:
        raise type(e)(
            f'failed to get_target_modules, due to: {e}. Please check the modules specified in --lora_unet_blocks are correct'
            )
