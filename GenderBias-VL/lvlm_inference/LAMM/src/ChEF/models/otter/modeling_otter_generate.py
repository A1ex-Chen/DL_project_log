@torch.no_grad()
def generate(self, vision_x: torch.Tensor, lang_x: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None, **generate_kwargs):
    """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
    if hasattr(self, '_hf_hook'):
        hook = AlignDevicesHook(execution_device=lang_x.device,
            io_same_device=True, place_submodules=False)
        add_hook_to_module(self.lang_encoder, hook)
    num_beams = generate_kwargs.get('num_beams', 1)
    if num_beams > 1:
        vision_x = vision_x.repeat_interleave(num_beams, dim=0)
    self._encode_vision_x(vision_x=vision_x)
    output = self.lang_encoder.generate(lang_x, attention_mask=
        attention_mask, eos_token_id=self.eoc_token_id, **generate_kwargs)
    self.lang_encoder.clear_conditioned_layers()
    return output
