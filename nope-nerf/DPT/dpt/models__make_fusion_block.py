def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False,
        bn=use_bn, expand=False, align_corners=True)
