def get_processor(self, return_deprecated_lora: bool=False
    ) ->'AttentionProcessor':
    """
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
    if not return_deprecated_lora:
        return self.processor
    is_lora_activated = {name: (module.lora_layer is not None) for name,
        module in self.named_modules() if hasattr(module, 'lora_layer')}
    if not any(is_lora_activated.values()):
        return self.processor
    is_lora_activated.pop('add_k_proj', None)
    is_lora_activated.pop('add_v_proj', None)
    if not all(is_lora_activated.values()):
        raise ValueError(
            f'Make sure that either all layers or no layers have LoRA activated, but have {is_lora_activated}'
            )
    non_lora_processor_cls_name = self.processor.__class__.__name__
    lora_processor_cls = getattr(import_module(__name__), 'LoRA' +
        non_lora_processor_cls_name)
    hidden_size = self.inner_dim
    if lora_processor_cls in [LoRAAttnProcessor, LoRAAttnProcessor2_0,
        LoRAXFormersAttnProcessor]:
        kwargs = {'cross_attention_dim': self.cross_attention_dim, 'rank':
            self.to_q.lora_layer.rank, 'network_alpha': self.to_q.
            lora_layer.network_alpha, 'q_rank': self.to_q.lora_layer.rank,
            'q_hidden_size': self.to_q.lora_layer.out_features, 'k_rank':
            self.to_k.lora_layer.rank, 'k_hidden_size': self.to_k.
            lora_layer.out_features, 'v_rank': self.to_v.lora_layer.rank,
            'v_hidden_size': self.to_v.lora_layer.out_features, 'out_rank':
            self.to_out[0].lora_layer.rank, 'out_hidden_size': self.to_out[
            0].lora_layer.out_features}
        if hasattr(self.processor, 'attention_op'):
            kwargs['attention_op'] = self.processor.attention_op
        lora_processor = lora_processor_cls(hidden_size, **kwargs)
        lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.
            state_dict())
        lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.
            state_dict())
        lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.
            state_dict())
        lora_processor.to_out_lora.load_state_dict(self.to_out[0].
            lora_layer.state_dict())
    elif lora_processor_cls == LoRAAttnAddedKVProcessor:
        lora_processor = lora_processor_cls(hidden_size,
            cross_attention_dim=self.add_k_proj.weight.shape[0], rank=self.
            to_q.lora_layer.rank, network_alpha=self.to_q.lora_layer.
            network_alpha)
        lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.
            state_dict())
        lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.
            state_dict())
        lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.
            state_dict())
        lora_processor.to_out_lora.load_state_dict(self.to_out[0].
            lora_layer.state_dict())
        if self.add_k_proj.lora_layer is not None:
            lora_processor.add_k_proj_lora.load_state_dict(self.add_k_proj.
                lora_layer.state_dict())
            lora_processor.add_v_proj_lora.load_state_dict(self.add_v_proj.
                lora_layer.state_dict())
        else:
            lora_processor.add_k_proj_lora = None
            lora_processor.add_v_proj_lora = None
    else:
        raise ValueError(f'{lora_processor_cls} does not exist.')
    return lora_processor
