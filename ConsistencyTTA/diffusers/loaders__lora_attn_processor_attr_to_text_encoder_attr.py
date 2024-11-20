@property
def _lora_attn_processor_attr_to_text_encoder_attr(self):
    return {'to_q_lora': 'q_proj', 'to_k_lora': 'k_proj', 'to_v_lora':
        'v_proj', 'to_out_lora': 'out_proj'}
