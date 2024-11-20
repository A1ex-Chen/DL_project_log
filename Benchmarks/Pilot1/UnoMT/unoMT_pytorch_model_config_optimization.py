def config_optimization(self):
    args = self.args
    self.update_l2regularizer(args.l2_regularization)
    self.resp_lr_decay = LambdaLR(optimizer=self.resp_opt, lr_lambda=lambda
        e: args.lr_decay_factor ** e)
    self.cl_clf_lr_decay = LambdaLR(optimizer=self.cl_clf_opt, lr_lambda=lambda
        e: args.lr_decay_factor ** e)
    self.drug_target_lr_decay = LambdaLR(optimizer=self.drug_target_opt,
        lr_lambda=lambda e: args.lr_decay_factor ** e)
    self.drug_qed_lr_decay = LambdaLR(optimizer=self.drug_qed_opt,
        lr_lambda=lambda e: args.lr_decay_factor ** e)
    self.resp_loss_func = (F.l1_loss if args.resp_loss_func == 'l1' else F.
        mse_loss)
    self.drug_qed_loss_func = (F.l1_loss if args.drug_qed_loss_func == 'l1'
         else F.mse_loss)
