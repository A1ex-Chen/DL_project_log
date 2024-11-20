def forward(self, x):
    if rms_norm is not None and x.is_cuda:
        return rms_norm(x, self.weight, self.eps)
    else:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
