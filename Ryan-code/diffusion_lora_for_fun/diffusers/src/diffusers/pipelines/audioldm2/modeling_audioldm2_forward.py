def forward(self, hidden_states: torch.Tensor, res_hidden_states_tuple:
    Tuple[torch.Tensor, ...], temb: Optional[torch.Tensor]=None,
    encoder_hidden_states: Optional[torch.Tensor]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, upsample_size:
    Optional[int]=None, attention_mask: Optional[torch.Tensor]=None,
    encoder_attention_mask: Optional[torch.Tensor]=None,
    encoder_hidden_states_1: Optional[torch.Tensor]=None,
    encoder_attention_mask_1: Optional[torch.Tensor]=None):
    num_layers = len(self.resnets)
    num_attention_per_layer = len(self.attentions) // num_layers
    encoder_hidden_states_1 = (encoder_hidden_states_1 if 
        encoder_hidden_states_1 is not None else encoder_hidden_states)
    encoder_attention_mask_1 = (encoder_attention_mask_1 if 
        encoder_hidden_states_1 is not None else encoder_attention_mask)
    for i in range(num_layers):
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
                create_custom_forward(self.resnets[i]), hidden_states, temb,
                **ckpt_kwargs)
            for idx, cross_attention_dim in enumerate(self.cross_attention_dim
                ):
                if cross_attention_dim is not None and idx <= 1:
                    forward_encoder_hidden_states = encoder_hidden_states
                    forward_encoder_attention_mask = encoder_attention_mask
                elif cross_attention_dim is not None and idx > 1:
                    forward_encoder_hidden_states = encoder_hidden_states_1
                    forward_encoder_attention_mask = encoder_attention_mask_1
                else:
                    forward_encoder_hidden_states = None
                    forward_encoder_attention_mask = None
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.attentions[i *
                    num_attention_per_layer + idx], return_dict=False),
                    hidden_states, forward_encoder_hidden_states, None,
                    None, cross_attention_kwargs, attention_mask,
                    forward_encoder_attention_mask, **ckpt_kwargs)[0]
        else:
            hidden_states = self.resnets[i](hidden_states, temb)
            for idx, cross_attention_dim in enumerate(self.cross_attention_dim
                ):
                if cross_attention_dim is not None and idx <= 1:
                    forward_encoder_hidden_states = encoder_hidden_states
                    forward_encoder_attention_mask = encoder_attention_mask
                elif cross_attention_dim is not None and idx > 1:
                    forward_encoder_hidden_states = encoder_hidden_states_1
                    forward_encoder_attention_mask = encoder_attention_mask_1
                else:
                    forward_encoder_hidden_states = None
                    forward_encoder_attention_mask = None
                hidden_states = self.attentions[i * num_attention_per_layer +
                    idx](hidden_states, attention_mask=attention_mask,
                    encoder_hidden_states=forward_encoder_hidden_states,
                    encoder_attention_mask=forward_encoder_attention_mask,
                    return_dict=False)[0]
    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
    return hidden_states
