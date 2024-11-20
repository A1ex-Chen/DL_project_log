def create_optimizer_and_scheduler(self, num_training_steps: int):
    """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
    if self.optimizer is None:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.
            model.named_parameters() if not any(nd in n for nd in no_decay)
            ], 'weight_decay': self.args.weight_decay}, {'params': [p for n,
            p in self.model.named_parameters() if any(nd in n for nd in
            no_decay)], 'weight_decay': 0.0}]
        if self.args.adafactor:
            self.optimizer = Adafactor(optimizer_grouped_parameters, lr=
                self.args.learning_rate, scale_parameter=False,
                relative_step=False)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.
                args.learning_rate, eps=self.args.adam_epsilon)
    if self.lr_scheduler is None:
        self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
    else:
        logger.warn(
            'scheduler is passed to `Seq2SeqTrainer`, `--lr_scheduler` arg is ignored.'
            )
