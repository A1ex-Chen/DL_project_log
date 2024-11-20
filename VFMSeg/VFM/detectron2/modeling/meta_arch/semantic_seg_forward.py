def forward(self, features, targets=None):
    """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
    x = self.layers(features)
    if self.training:
        return None, self.losses(x, targets)
    else:
        x = F.interpolate(x, scale_factor=self.common_stride, mode=
            'bilinear', align_corners=False)
        return x, {}
