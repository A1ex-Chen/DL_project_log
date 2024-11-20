def prepare_inputs_for_generation(self, decoder_input_ids, past=None,
    attention_mask=None, use_cache=None, encoder_outputs=None, doc_scores=
    None, n_docs=None, **kwargs):
    return {'input_ids': None, 'encoder_outputs': encoder_outputs,
        'doc_scores': doc_scores, 'context_attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids, 'past_key_values': past,
        'use_cache': use_cache, 'do_marginalize': True, 'n_docs': n_docs}
