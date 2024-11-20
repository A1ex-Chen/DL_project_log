def create_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, lm_labels
    ):
    model = TransfoXLLMHeadModel(config)
    model.eval()
    lm_logits_1, mems_1 = model(input_ids_1)
    loss_1, _, mems_1 = model(input_ids_1, labels=lm_labels)
    lm_logits_2, mems_2 = model(input_ids_2, mems=mems_1)
    loss_2, _, mems_2 = model(input_ids_2, labels=lm_labels, mems=mems_1)
    outputs = {'loss_1': loss_1, 'mems_1': mems_1, 'lm_logits_1':
        lm_logits_1, 'loss_2': loss_2, 'mems_2': mems_2, 'lm_logits_2':
        lm_logits_2}
    return outputs
