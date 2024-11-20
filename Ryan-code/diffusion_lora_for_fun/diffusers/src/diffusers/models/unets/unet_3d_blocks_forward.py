def forward(self, hidden_states: torch.Tensor, res_hidden_states_tuple:
    Tuple[torch.Tensor, ...], temb: Optional[torch.Tensor]=None,
    encoder_hidden_states: Optional[torch.Tensor]=None,
    image_only_indicator: Optional[torch.Tensor]=None) ->torch.Tensor:
    for resnet, attn in zip(self.resnets, self.attentions):
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet), hidden_states, temb,
                image_only_indicator, **ckpt_kwargs)
            hidden_states = attn(hidden_states, encoder_hidden_states=
                encoder_hidden_states, image_only_indicator=
                image_only_indicator, return_dict=False)[0]
        else:
            hidden_states = resnet(hidden_states, temb,
                image_only_indicator=image_only_indicator)
            hidden_states = attn(hidden_states, encoder_hidden_states=
                encoder_hidden_states, image_only_indicator=
                image_only_indicator, return_dict=False)[0]
    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
    return hidden_states
