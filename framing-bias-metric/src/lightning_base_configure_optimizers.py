def configure_optimizers(self):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    model = self.model
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.
        named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': self.hparams.weight_decay}, {'params': [p for n, p in
        model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}]
    if self.hparams.adafactor:
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams
            .learning_rate, scale_parameter=False, relative_step=False)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.
            learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    scheduler = self.get_lr_scheduler()
    return [optimizer], [scheduler]
