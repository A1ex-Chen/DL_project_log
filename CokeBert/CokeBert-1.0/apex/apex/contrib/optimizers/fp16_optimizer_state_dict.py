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
    state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
    state_dict['cur_scale'] = self.cur_scale
    state_dict['cur_iter'] = self.cur_iter
    if state_dict['dynamic_loss_scale']:
        state_dict['last_overflow_iter'] = self.last_overflow_iter
        state_dict['scale_factor'] = self.scale_factor
        state_dict['scale_window'] = self.scale_window
    state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
    state_dict['fp32_groups_flat'] = self.fp32_groups_flat
    return state_dict
