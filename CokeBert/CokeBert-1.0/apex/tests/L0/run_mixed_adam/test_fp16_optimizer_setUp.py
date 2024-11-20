def setUp(self, max_abs_diff=0.001, max_rel_diff=1, iters=7):
    self.max_abs_diff = max_abs_diff
    self.max_rel_diff = max_rel_diff
    self.iters = iters
    torch.cuda.manual_seed(13337)
    N, D_in, D_out = 64, 1024, 16
    self.N = N
    self.D_in = D_in
    self.D_out = D_out
    self.x = torch.randn((N, D_in), dtype=torch.float16, device='cuda')
    self.ref_model = torch.nn.Linear(D_in, D_out).cuda().half()
    self.tst_model = torch.nn.Linear(D_in, D_out).cuda().half()
    for p, q in zip(self.tst_model.parameters(), self.ref_model.parameters()):
        p.data.copy_(q.data)
