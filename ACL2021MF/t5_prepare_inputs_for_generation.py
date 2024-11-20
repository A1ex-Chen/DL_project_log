def prepare_inputs_for_generation(self, input_ids, past=None,
    attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
    history_ids = input_ids.clone()
    if past is not None:
        input_ids = input_ids[:, -1:]
    generation = {'decoder_input_ids': input_ids, 'past_key_values': past,
        'encoder_outputs': encoder_outputs, 'attention_mask':
        attention_mask, 'use_cache': use_cache, 'decoder_history_input_ids':
        history_ids}
    for kwarg in ['decoder_copy_pos', 'decoder_concept_cls',
        'decoder_mention_flag', 'decoder_copy_mention_flag',
        'decoder_cls_on_input', 'encoder_img_mask', 'encoder_obj_feature',
        'encoder_obj_box', 'encoder_relative_pos_index']:
        if kwarg in kwargs:
            generation[kwarg] = kwargs[kwarg]
    return generation
