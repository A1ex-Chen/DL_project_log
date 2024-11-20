def _step(self, batch: dict) ->Tuple:
    pad_token_id = self.tokenizer.pad_token_id
    src_ids, src_mask = batch['input_ids'], batch['attention_mask']
    tgt_ids = batch['labels']
    if isinstance(self.model, T5ForConditionalGeneration):
        decoder_input_ids = self.model._shift_right(tgt_ids)
    else:
        decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
    if not self.already_saved_batch:
        batch['decoder_input_ids'] = decoder_input_ids
        self.save_readable_batch(batch)
    if self.mode == 'summarization':
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=
            decoder_input_ids, use_cache=False)
        lm_logits = outputs['logits']
        if self.hparams.label_smoothing == 0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]),
                tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(lprobs, tgt_ids, self.
                hparams.label_smoothing, ignore_index=pad_token_id)
        self.log('lm_loss', loss, on_step=True, on_epoch=False, prog_bar=
            True, logger=True)
    elif self.mode == 'neutralizationMT':
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=
            decoder_input_ids, use_cache=False, extra_task=self.extra_task,
            extra_task_input_ids=batch['extra_task_text_input_ids'] if 
            'extra_task_text_input_ids' in batch.keys() else None,
            extra_task_attention_mask=batch[
            'extra_task_batch_attention_mask'] if 
            'extra_task_batch_attention_mask' in batch.keys() else None,
            extra_task_label=batch['extra_task_labels'] if 
            'extra_task_labels' in batch.keys() else None)
        lm_logits = outputs['logits']
        extra_task_loss = outputs['loss'] if 'loss' in outputs else None
        if self.hparams.label_smoothing == 0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]),
                tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(lprobs, tgt_ids, self.
                hparams.label_smoothing, ignore_index=pad_token_id)
        self.log('lm_loss', loss, on_step=True, on_epoch=False, prog_bar=
            True, logger=True)
        if extra_task_loss != None:
            self.log('e_loss', extra_task_loss, on_step=True, on_epoch=
                False, prog_bar=True, logger=True)
            loss = loss + self.task_loss_ratio * extra_task_loss
            self.log('t_loss', loss, on_step=True, on_epoch=False, prog_bar
                =True, logger=True)
    return loss,
