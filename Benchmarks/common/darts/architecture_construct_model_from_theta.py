def construct_model_from_theta(self, theta):
    """
        construct a new model with initialized weight from theta
        it use .state_dict() and load_state_dict() instead of
        .parameters() + fill_()
        :param theta: flatten weights, need to reshape to original shape
        :return:
        """
    model = self.model.new()
    state_dict = self.model.state_dict()
    params, offset = {}, 0
    for k, v in self.model.named_parameters():
        v_length = v.numel()
        params[k] = theta[offset:offset + v_length].view(v.size())
        offset += v_length
    assert offset == len(theta)
    state_dict.update(params)
    model.load_state_dict(state_dict)
    model.to(self.device)
    return model
