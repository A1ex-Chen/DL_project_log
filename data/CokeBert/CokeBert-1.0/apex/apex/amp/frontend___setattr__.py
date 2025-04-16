def __setattr__(self, name, value):
    if 'options' in self.__dict__:
        if name in self.options:
            if name == 'cast_model_type':
                if self.opt_level == 'O1' and value is not None:
                    if value is not False:
                        if value is not torch.float32:
                            warn_or_err(
                                "O1 inserts casts around Torch functions rather than model weights, so with O1, the model weights themselves should remain FP32. If you wish to cast the model to a different type, use opt_level='O2' or 'O3'. "
                                 + 'cast_model_type was {}'.format(value))
                self.options[name] = value
            elif name == 'patch_torch_functions':
                if self.opt_level != 'O1' and value:
                    warn_or_err(
                        "Currently, patch_torch_functions=True should only be set by selecting opt_level='O1'."
                        )
                self.options[name] = value
            elif name == 'keep_batchnorm_fp32':
                if self.opt_level == 'O1' and value is not None:
                    warn_or_err(
                        'With opt_level O1, batchnorm functions are automatically patched to run in FP32, so keep_batchnorm_fp32 should be None.'
                         + ' keep_batchnorm_fp32 was {}'.format(value))
                if value == 'False':
                    self.options[name] = False
                elif value == 'True':
                    self.options[name] = True
                else:
                    assert value is True or value is False or value is None, "keep_batchnorm_fp32 must be a boolean, the string 'True' or 'False', or None, found keep_batchnorm_fp32={}".format(
                        value)
                    self.options[name] = value
            elif name == 'master_weights':
                if self.opt_level == 'O1' and value is not None:
                    warn_or_err(
                        "It doesn't make sense to use master_weights with O1. With O1, your model weights themselves should be FP32."
                        )
                self.options[name] = value
            elif name == 'loss_scale':
                if value == 'dynamic':
                    self.options[name] = value
                else:
                    self.options[name] = float(value)
            else:
                self.options[name] = value
    else:
        super(Properties, self).__setattr__(name, value)
