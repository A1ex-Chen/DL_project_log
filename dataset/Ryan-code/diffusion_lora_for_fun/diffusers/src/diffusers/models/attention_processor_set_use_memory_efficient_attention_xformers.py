def set_use_memory_efficient_attention_xformers(self,
    use_memory_efficient_attention_xformers: bool, attention_op: Optional[
    Callable]=None) ->None:
    """
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
    is_lora = hasattr(self, 'processor') and isinstance(self.processor,
        LORA_ATTENTION_PROCESSORS)
    is_custom_diffusion = hasattr(self, 'processor') and isinstance(self.
        processor, (CustomDiffusionAttnProcessor,
        CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0))
    is_added_kv_processor = hasattr(self, 'processor') and isinstance(self.
        processor, (AttnAddedKVProcessor, AttnAddedKVProcessor2_0,
        SlicedAttnAddedKVProcessor, XFormersAttnAddedKVProcessor,
        LoRAAttnAddedKVProcessor))
    if use_memory_efficient_attention_xformers:
        if is_added_kv_processor and (is_lora or is_custom_diffusion):
            raise NotImplementedError(
                f'Memory efficient attention is currently not supported for LoRA or custom diffusion for attention processor type {self.processor}'
                )
        if not is_xformers_available():
            raise ModuleNotFoundError(
                'Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers'
                , name='xformers')
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only available for GPU "
                )
        else:
            try:
                _ = xformers.ops.memory_efficient_attention(torch.randn((1,
                    2, 40), device='cuda'), torch.randn((1, 2, 40), device=
                    'cuda'), torch.randn((1, 2, 40), device='cuda'))
            except Exception as e:
                raise e
        if is_lora:
            processor = LoRAXFormersAttnProcessor(hidden_size=self.
                processor.hidden_size, cross_attention_dim=self.processor.
                cross_attention_dim, rank=self.processor.rank, attention_op
                =attention_op)
            processor.load_state_dict(self.processor.state_dict())
            processor.to(self.processor.to_q_lora.up.weight.device)
        elif is_custom_diffusion:
            processor = CustomDiffusionXFormersAttnProcessor(train_kv=self.
                processor.train_kv, train_q_out=self.processor.train_q_out,
                hidden_size=self.processor.hidden_size, cross_attention_dim
                =self.processor.cross_attention_dim, attention_op=attention_op)
            processor.load_state_dict(self.processor.state_dict())
            if hasattr(self.processor, 'to_k_custom_diffusion'):
                processor.to(self.processor.to_k_custom_diffusion.weight.device
                    )
        elif is_added_kv_processor:
            logger.info(
                'Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation.'
                )
            processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
        else:
            processor = XFormersAttnProcessor(attention_op=attention_op)
    elif is_lora:
        attn_processor_class = LoRAAttnProcessor2_0 if hasattr(F,
            'scaled_dot_product_attention') else LoRAAttnProcessor
        processor = attn_processor_class(hidden_size=self.processor.
            hidden_size, cross_attention_dim=self.processor.
            cross_attention_dim, rank=self.processor.rank)
        processor.load_state_dict(self.processor.state_dict())
        processor.to(self.processor.to_q_lora.up.weight.device)
    elif is_custom_diffusion:
        attn_processor_class = CustomDiffusionAttnProcessor2_0 if hasattr(F,
            'scaled_dot_product_attention') else CustomDiffusionAttnProcessor
        processor = attn_processor_class(train_kv=self.processor.train_kv,
            train_q_out=self.processor.train_q_out, hidden_size=self.
            processor.hidden_size, cross_attention_dim=self.processor.
            cross_attention_dim)
        processor.load_state_dict(self.processor.state_dict())
        if hasattr(self.processor, 'to_k_custom_diffusion'):
            processor.to(self.processor.to_k_custom_diffusion.weight.device)
    else:
        processor = AttnProcessor2_0() if hasattr(F,
            'scaled_dot_product_attention'
            ) and self.scale_qk else AttnProcessor()
    self.set_processor(processor)
