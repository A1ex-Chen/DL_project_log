def to_hyper_param_dict(self) ->Dict[str, Union[int, float, str, bool, Tensor]
    ]:
    hyper_param_dict = super().to_hyper_param_dict()
    hyper_param_dict.update({'algorithm_name': str(self.algorithm_name),
        'pretrained': self.pretrained, 'num_frozen_levels': self.
        num_frozen_levels, 'eval_center_crop_ratio': self.
        eval_center_crop_ratio})
    return hyper_param_dict
