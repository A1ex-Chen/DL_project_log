def set_parameters(self, parameters):
    params_dict = zip(self.model.state_dict().keys(), parameters)
    state_dict = OrderedDict[{k: torch.tensor(v) for k, v in params_dict}]
    self.model.load_state_dict(state_dict, strict=True)
