def setUp(self):
    self.module_cpu_ = apex.normalization.FusedLayerNorm(normalized_shape=[
        32, 16], elementwise_affine=True).cpu()
    self.module_cuda_ = apex.normalization.FusedLayerNorm(normalized_shape=
        [32, 16], elementwise_affine=True).cuda()
