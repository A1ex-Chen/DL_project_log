def _compute_loss(self, model, inputs, labels):
    if self.args.label_smoothing == 0:
        if (self.data_args is not None and self.data_args.
            ignore_pad_token_for_loss):
            logits = model(**inputs, use_cache=False)[0]
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.
                view(-1))
        else:
            loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
    else:
        logits = model(**inputs, use_cache=False)[0]
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing,
            ignore_index=self.config.pad_token_id)
    return loss, logits
