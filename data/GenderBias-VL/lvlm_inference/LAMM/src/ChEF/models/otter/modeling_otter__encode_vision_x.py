def _encode_vision_x(self, vision_x: torch.Tensor):
    """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
    assert vision_x.ndim == 6, 'vision_x should be of shape (b, T_img, F, C, H, W)'
    b, T, F = vision_x.shape[:3]
    vision_x = rearrange(vision_x, 'b T F c h w -> (b T F) c h w')
    with torch.no_grad():
        vision_x = self.vision_encoder(vision_x)[0][:, 1:, :]
    vision_x = rearrange(vision_x, '(b T F) v d -> b T F v d', b=b, T=T, F=F)
    dtype = self.lang_encoder.lm_head.weight.dtype
    vision_x = self.perceiver(vision_x.to(self.lang_encoder.device, dtype=
        dtype))
    for layer in self.lang_encoder._get_decoder_layers():
        layer.condition_vis_x(vision_x)
