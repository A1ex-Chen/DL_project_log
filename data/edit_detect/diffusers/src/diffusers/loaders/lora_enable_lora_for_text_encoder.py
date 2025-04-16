def enable_lora_for_text_encoder(self, text_encoder: Optional[
    'PreTrainedModel']=None):
    """
        Enables the LoRA layers for the text encoder.

        Args:
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to enable the LoRA layers for. If `None`, it will try to get the `text_encoder`
                attribute.
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    text_encoder = text_encoder or getattr(self, 'text_encoder', None)
    if text_encoder is None:
        raise ValueError('Text Encoder not found.')
    set_adapter_layers(self.text_encoder, enabled=True)
