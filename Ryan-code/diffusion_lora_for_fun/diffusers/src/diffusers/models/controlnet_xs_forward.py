def forward(self, hidden_states: Tensor, res_hidden_states_tuple_base:
    Tuple[Tensor, ...], res_hidden_states_tuple_ctrl: Tuple[Tensor, ...],
    temb: Tensor, encoder_hidden_states: Optional[Tensor]=None,
    conditioning_scale: Optional[float]=1.0, cross_attention_kwargs:
    Optional[Dict[str, Any]]=None, attention_mask: Optional[Tensor]=None,
    upsample_size: Optional[int]=None, encoder_attention_mask: Optional[
    Tensor]=None, apply_control: bool=True) ->Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get('scale', None) is not None:
            logger.warning(
                'Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.'
                )
    is_freeu_enabled = getattr(self, 's1', None) and getattr(self, 's2', None
        ) and getattr(self, 'b1', None) and getattr(self, 'b2', None)

    def create_custom_forward(module, return_dict=None):

        def custom_forward(*inputs):
            if return_dict is not None:
                return module(*inputs, return_dict=return_dict)
            else:
                return module(*inputs)
        return custom_forward

    def maybe_apply_freeu_to_subblock(hidden_states, res_h_base):
        if is_freeu_enabled:
            return apply_freeu(self.resolution_idx, hidden_states,
                res_h_base, s1=self.s1, s2=self.s2, b1=self.b1, b2=self.b2)
        else:
            return hidden_states, res_h_base
    for resnet, attn, c2b, res_h_base, res_h_ctrl in zip(self.resnets, self
        .attentions, self.ctrl_to_base, reversed(
        res_hidden_states_tuple_base), reversed(res_hidden_states_tuple_ctrl)):
        if apply_control:
            hidden_states += c2b(res_h_ctrl) * conditioning_scale
        hidden_states, res_h_base = maybe_apply_freeu_to_subblock(hidden_states
            , res_h_base)
        hidden_states = torch.cat([hidden_states, res_h_base], dim=1)
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet), hidden_states, temb, **
                ckpt_kwargs)
        else:
            hidden_states = resnet(hidden_states, temb)
        if attn is not None:
            hidden_states = attn(hidden_states, encoder_hidden_states=
                encoder_hidden_states, cross_attention_kwargs=
                cross_attention_kwargs, attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask, return_dict=
                False)[0]
    if self.upsamplers is not None:
        hidden_states = self.upsamplers(hidden_states, upsample_size)
    return hidden_states
