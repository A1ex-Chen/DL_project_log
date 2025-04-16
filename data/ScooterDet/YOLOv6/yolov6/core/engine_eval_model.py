def eval_model(self):
    if not hasattr(self.cfg, 'eval_params'):
        results, vis_outputs, vis_paths = eval.run(self.data_dict,
            batch_size=self.batch_size // self.world_size * 2, img_size=
            self.img_size, model=self.ema.ema if self.args.calib is False else
            self.model, conf_thres=0.03, dataloader=self.val_loader,
            save_dir=self.save_dir, task='train', specific_shape=self.
            specific_shape, height=self.height, width=self.width)
    else:

        def get_cfg_value(cfg_dict, value_str, default_value):
            if value_str in cfg_dict:
                if isinstance(cfg_dict[value_str], list):
                    return cfg_dict[value_str][0] if cfg_dict[value_str][0
                        ] is not None else default_value
                else:
                    return cfg_dict[value_str] if cfg_dict[value_str
                        ] is not None else default_value
            else:
                return default_value
        eval_img_size = get_cfg_value(self.cfg.eval_params, 'img_size',
            self.img_size)
        results, vis_outputs, vis_paths = eval.run(self.data_dict,
            batch_size=get_cfg_value(self.cfg.eval_params, 'batch_size', 
            self.batch_size // self.world_size * 2), img_size=eval_img_size,
            model=self.ema.ema if self.args.calib is False else self.model,
            conf_thres=get_cfg_value(self.cfg.eval_params, 'conf_thres', 
            0.03), dataloader=self.val_loader, save_dir=self.save_dir, task
            ='train', shrink_size=get_cfg_value(self.cfg.eval_params,
            'shrink_size', eval_img_size), infer_on_rect=get_cfg_value(self
            .cfg.eval_params, 'infer_on_rect', False), verbose=
            get_cfg_value(self.cfg.eval_params, 'verbose', False),
            do_coco_metric=get_cfg_value(self.cfg.eval_params,
            'do_coco_metric', True), do_pr_metric=get_cfg_value(self.cfg.
            eval_params, 'do_pr_metric', False), plot_curve=get_cfg_value(
            self.cfg.eval_params, 'plot_curve', False),
            plot_confusion_matrix=get_cfg_value(self.cfg.eval_params,
            'plot_confusion_matrix', False), specific_shape=self.
            specific_shape, height=self.height, width=self.width)
    LOGGER.info(
        f'Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}'
        )
    self.evaluate_results = results[:2]
    self.plot_val_pred(vis_outputs, vis_paths)
