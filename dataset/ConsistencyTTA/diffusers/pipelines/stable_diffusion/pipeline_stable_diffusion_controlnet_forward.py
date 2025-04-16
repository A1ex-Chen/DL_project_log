def forward(self, sample: torch.FloatTensor, timestep: Union[torch.Tensor,
    float, int], encoder_hidden_states: torch.Tensor, controlnet_cond: List
    [torch.tensor], conditioning_scale: List[float], class_labels: Optional
    [torch.Tensor]=None, timestep_cond: Optional[torch.Tensor]=None,
    attention_mask: Optional[torch.Tensor]=None, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, return_dict: bool=True) ->Union[
    ControlNetOutput, Tuple]:
    for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond,
        conditioning_scale, self.nets)):
        down_samples, mid_sample = controlnet(sample, timestep,
            encoder_hidden_states, image, scale, class_labels,
            timestep_cond, attention_mask, cross_attention_kwargs, return_dict)
        if i == 0:
            down_block_res_samples, mid_block_res_sample = (down_samples,
                mid_sample)
        else:
            down_block_res_samples = [(samples_prev + samples_curr) for 
                samples_prev, samples_curr in zip(down_block_res_samples,
                down_samples)]
            mid_block_res_sample += mid_sample
    return down_block_res_samples, mid_block_res_sample
