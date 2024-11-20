def train(self):
    args = self.args
    device = self.device
    self.val_cl_clf_acc = []
    self.val_drug_target_acc = []
    self.val_drug_qed_mse = []
    self.val_drug_qed_mae = []
    self.val_drug_qed_r2 = []
    self.val_resp_mse = []
    self.val_resp_mae = []
    self.val_resp_r2 = []
    self.best_r2 = -np.inf
    self.patience = 0
    self.start_time = time.time()
    for epoch in range(args.epochs):
        print('=' * 80 + '\nTraining Epoch %3i:' % (epoch + 1))
        epoch_start_time = time.time()
        self.resp_lr_decay.step(epoch)
        self.cl_clf_lr_decay.step(epoch)
        self.drug_target_lr_decay.step(epoch)
        self.drug_qed_lr_decay.step(epoch)
        train_cl_clf(device=device, category_clf_net=self.category_clf_net,
            site_clf_net=self.site_clf_net, type_clf_net=self.type_clf_net,
            data_loader=self.cl_clf_trn_loader, max_num_batches=args.
            max_num_batches, optimizer=self.cl_clf_opt)
        train_drug_target(device=device, drug_target_net=self.
            drug_target_net, data_loader=self.drug_target_trn_loader,
            max_num_batches=args.max_num_batches, optimizer=self.
            drug_target_opt)
        train_drug_qed(device=device, drug_qed_net=self.drug_qed_net,
            data_loader=self.drug_qed_trn_loader, max_num_batches=args.
            max_num_batches, loss_func=self.drug_qed_loss_func, optimizer=
            self.drug_qed_opt)
        train_resp(device=device, resp_net=self.resp_net, data_loader=self.
            drug_resp_trn_loader, max_num_batches=args.max_num_batches,
            loss_func=self.resp_loss_func, optimizer=self.resp_opt)
        if epoch >= args.resp_val_start_epoch:
            resp_r2 = self.validation(epoch)
            if resp_r2[self.val_index] > self.best_r2:
                self.patience = 0
                self.best_r2 = resp_r2[self.val_index]
            else:
                self.patience += 1
            if self.patience >= args.early_stop_patience:
                print(
                    'Validation results does not improve for %d epochs ... invoking early stopping.'
                     % self.patience)
                break
        print('Epoch Running Time: %.1f Seconds.' % (time.time() -
            epoch_start_time))
