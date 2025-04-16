def test_cpu_is_float(self):
    fn = lambda x: x.cpu()
    always_cpu_float = {torch.float: 'torch.FloatTensor', torch.half:
        'torch.FloatTensor'}
    run_layer_test(self, [fn], always_cpu_float, (self.b, self.h))
