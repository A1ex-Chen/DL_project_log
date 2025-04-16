def axpby(self, sizea, sizeb, applier, repeat_tensors, x_type, y_type,
    out_type, inplace=False):
    self.overflow_buf.zero_()
    t1 = torch.cuda.FloatTensor(sizea).fill_(1.0)
    t2 = torch.cuda.FloatTensor(sizeb).fill_(1.0)
    y_list = []
    for i in range(repeat_tensors):
        y_list += [t1.clone().to(y_type) * self.yval, t2.clone().to(y_type) *
            self.yval]
    x_list = [(x.clone().to(x_type) * (self.xval / self.yval)) for x in y_list]
    if inplace:
        out_list = y_list
    else:
        out_list = [(out.clone().to(out_type) * 3.0) for out in y_list]
    applier(multi_tensor_axpby, self.overflow_buf, [x_list, y_list,
        out_list], self.a, self.b, -1)
    self.assertTrue(all([torch.allclose(out, self.ref.to(out_type)) for out in
        out_list]), msg='{} {} {} {} {} {} {}'.format(sizea, sizeb,
        repeat_tensors, x_type, y_type, out_type, inplace))
    self.assertTrue(self.overflow_buf.item() == 0, msg=
        '{} {} {} {} {} {} {}'.format(sizea, sizeb, repeat_tensors, x_type,
        y_type, out_type, inplace))
