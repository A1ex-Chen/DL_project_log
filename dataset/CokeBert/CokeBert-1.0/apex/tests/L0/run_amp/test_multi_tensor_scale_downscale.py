def downscale(self, sizea, sizeb, applier, repeat_tensors, in_type,
    out_type, inplace=False):
    self.overflow_buf.zero_()
    a = torch.cuda.FloatTensor(sizea).fill_(self.scale)
    b = torch.cuda.FloatTensor(sizeb).fill_(self.scale)
    out_list = []
    for i in range(repeat_tensors):
        out_list += [a.clone().to(out_type), b.clone().to(out_type)]
    if inplace:
        in_list = out_list
    else:
        in_list = [out.clone().to(in_type) for out in out_list]
    applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1.0 /
        self.scale)
    self.assertTrue(all([torch.allclose(out, self.ref.to(out_type)) for out in
        out_list]))
    self.assertTrue(self.overflow_buf.item() == 0)
