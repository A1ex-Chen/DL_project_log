def y_exact(self, t):
    t = t.detach().cpu().numpy()
    A_np = self.A.detach().cpu().numpy()
    ans = []
    for t_i in t:
        ans.append(np.matmul(scipy.linalg.expm(A_np * t_i), self.initial_val))
    return torch.stack([torch.tensor(ans_) for ans_ in ans]).reshape(len(t),
        self.dim)
