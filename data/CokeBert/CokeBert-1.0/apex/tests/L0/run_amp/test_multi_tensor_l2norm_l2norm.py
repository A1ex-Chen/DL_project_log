def l2norm(self, sizea, sizeb, applier, repeat_tensors, in_type, per_tensor):
    self.overflow_buf.zero_()
    a = torch.cuda.FloatTensor(sizea).fill_(self.val)
    b = torch.cuda.FloatTensor(sizeb).fill_(self.val)
    in_list = []
    for i in range(repeat_tensors):
        in_list += [a.clone().to(in_type), b.clone().to(in_type)]
    if per_tensor:
        norm, norm_per_tensor = applier(multi_tensor_l2norm, self.
            overflow_buf, [in_list], True)
        normab = torch.cat((a.norm().view(1), b.norm().view(1)))
        norm_per_tensor = norm_per_tensor.view(-1, 2)
    else:
        norm, _ = applier(multi_tensor_l2norm, self.overflow_buf, [in_list],
            True)
    reference = torch.cuda.FloatTensor((sizea + sizeb) * repeat_tensors).fill_(
        self.val).norm()
    self.assertTrue(torch.allclose(norm, reference))
    if per_tensor:
        self.assertTrue(torch.allclose(norm_per_tensor, normab))
    self.assertTrue(self.overflow_buf.item() == 0)
