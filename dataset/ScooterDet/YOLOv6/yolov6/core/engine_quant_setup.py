def quant_setup(self, model, cfg, device):
    if self.args.quant:
        from tools.qat.qat_utils import qat_init_model_manu, skip_sensitive_layers
        qat_init_model_manu(model, cfg, self.args)
        model.neck.upsample_enable_quant(cfg.ptq.num_bits, cfg.ptq.calib_method
            )
        if self.args.calib is False:
            if cfg.qat.sensitive_layers_skip:
                skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)
            assert cfg.qat.calib_pt is not None, 'Please provide calibrated model'
            model.load_state_dict(torch.load(cfg.qat.calib_pt)['model'].
                float().state_dict())
        model.to(device)
