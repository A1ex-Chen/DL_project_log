def forward(self, input_ids=None, mems=None, head_mask=None, inputs_embeds=
    None, labels=None):
    if input_ids is not None:
        bsz, tgt_len = input_ids.size(0), input_ids.size(1)
    elif inputs_embeds is not None:
        bsz, tgt_len = inputs_embeds.size(0), inputs_embeds.size(1)
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')
    transformer_outputs = self.transformer(input_ids, mems=mems, head_mask=
        head_mask, inputs_embeds=inputs_embeds)
    last_hidden = transformer_outputs[0]
    pred_hid = last_hidden[:, -tgt_len:]
    outputs = transformer_outputs[1:]
    if self.sample_softmax > 0 and self.training:
        assert self.config.tie_weight
        logit = sample_logits(self.transformer.word_emb, self.out_layer.
            bias, labels, pred_hid, self.sampler)
        softmax_output = -F.log_softmax(logit, -1)[:, :, 0]
        outputs = [softmax_output] + outputs
        if labels is not None:
            raise NotImplementedError
    else:
        softmax_output = self.crit(pred_hid.view(-1, pred_hid.size(-1)), labels
            )
        if labels is None:
            softmax_output = softmax_output.view(bsz, tgt_len, -1)
            outputs = [softmax_output] + outputs
        else:
            softmax_output = softmax_output.view(bsz, tgt_len)
            outputs = [softmax_output, None] + outputs
    return outputs
