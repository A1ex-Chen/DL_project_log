def call(self, inputs, mems=None, head_mask=None, inputs_embeds=None,
    labels=None, training=False):
    if isinstance(inputs, (tuple, list)):
        input_ids = inputs[0]
        mems = inputs[1] if len(inputs) > 1 else mems
        head_mask = inputs[2] if len(inputs) > 2 else head_mask
        inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
        labels = inputs[4] if len(inputs) > 4 else labels
        assert len(inputs) <= 5, 'Too many inputs.'
    elif isinstance(inputs, dict):
        input_ids = inputs.get('input_ids')
        mems = inputs.get('mems', mems)
        head_mask = inputs.get('head_mask', head_mask)
        inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
        labels = inputs.get('labels', labels)
        assert len(inputs) <= 5, 'Too many inputs.'
    else:
        input_ids = inputs
    if input_ids is not None:
        bsz, tgt_len = shape_list(input_ids)[:2]
    else:
        bsz, tgt_len = shape_list(inputs_embeds)[:2]
    transformer_outputs = self.transformer([input_ids, mems, head_mask,
        inputs_embeds], training=training)
    last_hidden = transformer_outputs[0]
    pred_hid = last_hidden[:, -tgt_len:]
    outputs = transformer_outputs[1:]
    if self.sample_softmax > 0 and training:
        raise NotImplementedError
    else:
        softmax_output = self.crit([pred_hid, labels], training=training)
        outputs = [softmax_output] + outputs
    return outputs
