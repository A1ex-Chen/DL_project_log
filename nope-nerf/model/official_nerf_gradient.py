def gradient(self, p, it):
    with torch.enable_grad():
        p.requires_grad_(True)
        _, y = self.infer_occ(p)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y, inputs=p, grad_outputs=
            d_output, create_graph=True, retain_graph=True, only_inputs=
            True, allow_unused=True)[0]
        return -gradients.unsqueeze(1)
