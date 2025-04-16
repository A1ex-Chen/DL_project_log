def to_hyper_param_dict(self) ->Dict[str, Union[int, float, str, bool, Tensor]
    ]:
    hyper_param_dict = super().to_hyper_param_dict()
    hyper_param_dict.update({'algorithm_name': str(self.algorithm_name)})
    return hyper_param_dict
