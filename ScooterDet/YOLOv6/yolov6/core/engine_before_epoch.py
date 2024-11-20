def before_epoch(self):
    if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
        self.cfg.data_aug.mosaic = 0.0
        self.cfg.data_aug.mixup = 0.0
        self.args.cache_ram = False
        self.train_loader, self.val_loader = self.get_data_loader(self.args,
            self.cfg, self.data_dict)
    self.model.train()
    if self.rank != -1:
        self.train_loader.sampler.set_epoch(self.epoch)
    self.mean_loss = torch.zeros(self.loss_num, device=self.device)
    self.optimizer.zero_grad()
    LOGGER.info(('\n' + '%10s' * (self.loss_num + 2)) % (*self.loss_info,))
    self.pbar = enumerate(self.train_loader)
    if self.main_process:
        self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
