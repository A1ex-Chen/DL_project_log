def test_params_and_buffers(self):
    exempted_modules = [self.fp16_model.network.bn, self.fp16_model.network
        .dn.db1.bn, self.fp16_model.network.dn.db2.bn]
    for m in self.fp16_model.modules():
        expected_dtype = torch.float if m in exempted_modules else torch.half
        for p in m.parameters(recurse=False):
            assert p.dtype == expected_dtype
        for b in m.buffers(recurse=False):
            assert b.dtype in (expected_dtype, torch.int64)
