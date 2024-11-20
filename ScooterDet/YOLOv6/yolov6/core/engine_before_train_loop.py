def before_train_loop(self):
    LOGGER.info('Training start...')
    self.start_time = time.time()
    self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.
        max_stepnum), 1000) if self.args.quant is False else 0
    self.scheduler.last_epoch = self.start_epoch - 1
    self.last_opt_step = -1
    self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
    self.best_ap, self.ap = 0.0, 0.0
    self.best_stop_strong_aug_ap = 0.0
    self.evaluate_results = 0, 0
    if hasattr(self, 'ckpt'):
        self.evaluate_results = self.ckpt['results']
        self.best_ap = self.evaluate_results[1]
        self.best_stop_strong_aug_ap = self.evaluate_results[1]
    self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
        ori_img_size=self.img_size, warmup_epoch=self.cfg.model.head.
        atss_warmup_epoch, use_dfl=self.cfg.model.head.use_dfl, reg_max=
        self.cfg.model.head.reg_max, iou_type=self.cfg.model.head.iou_type,
        fpn_strides=self.cfg.model.head.strides)
    if self.args.fuse_ab:
        self.compute_loss_ab = ComputeLoss_ab(num_classes=self.data_dict[
            'nc'], ori_img_size=self.img_size, warmup_epoch=0, use_dfl=
            False, reg_max=0, iou_type=self.cfg.model.head.iou_type,
            fpn_strides=self.cfg.model.head.strides)
    if self.args.distill:
        if self.cfg.model.type in ['YOLOv6n', 'YOLOv6s']:
            Loss_distill_func = ComputeLoss_distill_ns
        else:
            Loss_distill_func = ComputeLoss_distill
        self.compute_loss_distill = Loss_distill_func(num_classes=self.
            data_dict['nc'], ori_img_size=self.img_size, fpn_strides=self.
            cfg.model.head.strides, warmup_epoch=self.cfg.model.head.
            atss_warmup_epoch, use_dfl=self.cfg.model.head.use_dfl, reg_max
            =self.cfg.model.head.reg_max, iou_type=self.cfg.model.head.
            iou_type, distill_weight=self.cfg.model.head.distill_weight,
            distill_feat=self.args.distill_feat)
