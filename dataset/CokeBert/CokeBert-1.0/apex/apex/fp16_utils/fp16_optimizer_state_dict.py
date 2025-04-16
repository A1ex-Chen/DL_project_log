def state_dict(self):
    """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::

            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
    state_dict = {}
    state_dict['loss_scaler'] = self.loss_scaler
    state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
    state_dict['overflow'] = self.overflow
    state_dict['first_closure_call_this_step'
        ] = self.first_closure_call_this_step
    state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
    state_dict['fp32_from_fp16'] = self.fp32_from_fp16_groups
    return state_dict
