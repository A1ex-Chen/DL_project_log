@staticmethod
def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)
        ).rsqrt_()
    c_factor = exp_avg_sq_col.rsqrt()
    return torch.mm(r_factor.unsqueeze(-1), c_factor.unsqueeze(0))
