def calibrate(self, cfg):

    def save_calib_model(model, cfg):
        output_model_path = os.path.join(cfg.ptq.calib_output_path,
            '{}_calib_{}.pt'.format(os.path.splitext(os.path.basename(cfg.
            model.pretrained))[0], cfg.ptq.calib_method))
        if cfg.ptq.sensitive_layers_skip is True:
            output_model_path = output_model_path.replace('.pt', '_partial.pt')
        LOGGER.info('Saving calibrated model to {}... '.format(
            output_model_path))
        if not os.path.exists(cfg.ptq.calib_output_path):
            os.mkdir(cfg.ptq.calib_output_path)
        torch.save({'model': deepcopy(de_parallel(model)).half()},
            output_model_path)
    assert self.args.quant is True and self.args.calib is True
    if self.main_process:
        from tools.qat.qat_utils import ptq_calibrate
        ptq_calibrate(self.model, self.train_loader, cfg)
        self.epoch = 0
        self.eval_model()
        save_calib_model(self.model, cfg)