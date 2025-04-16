def make_batches(self, images, inputs):
    tokens, lengths, img_src_tokens, img_gpt_input_mask = (
        get_interactive_tokens_and_lengths(self.task, images, inputs, self.
        tokenizer, self.special_tokens))
    task = self.task
    cfg = self.cfg
    itr = task.get_batch_iterator(dataset=task.
        build_dataset_for_caption_inference(tokens, lengths, img_src_tokens,
        img_gpt_input_mask), max_sentences=cfg.dataset.batch_size,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test
        ).next_epoch_itr(shuffle=False)
    res_list = []
    for batch in itr:
        ids = batch['id']
        src_tokens = batch['net_input']['src_tokens'].to(self.device)
        src_lengths = batch['net_input']['src_lengths'].to(self.device)
        img_src_tokens = batch['net_input']['img_src_tokens'].to(dtype=self
            .dtype, device=self.device)
        img_gpt_input_mask = batch['net_input']['img_gpt_input_mask']
        res_list.append(dict(ids=ids, net_input=dict(src_tokens=src_tokens,
            src_lengths=src_lengths, img_src_tokens=img_src_tokens,
            img_gpt_input_mask=img_gpt_input_mask)))
    return res_list
