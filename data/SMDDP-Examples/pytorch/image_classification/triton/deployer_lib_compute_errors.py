def compute_errors(self, outputs_A, outputs_B):
    """ returns dictionary with errors statistics """
    device = outputs_A[0][0][0].device
    dtype = outputs_A[0][0][0].dtype
    x_values = torch.zeros(0, device=device, dtype=dtype)
    y_values = torch.zeros(0, device=device, dtype=dtype)
    d_values = torch.zeros(0, device=device, dtype=dtype)
    for output_A, output_B in zip(outputs_A, outputs_B):
        for x, y in zip(output_A, output_B):
            d = abs(x - y)
            x_values = torch.cat((x_values, x), 0)
            y_values = torch.cat((y_values, y), 0)
            d_values = torch.cat((d_values, d), 0)
    Error_stats = {'Original': self.compute_tensor_stats(x_values),
        'Converted': self.compute_tensor_stats(y_values),
        'Absolute difference': self.compute_tensor_stats(d_values)}
    return Error_stats
