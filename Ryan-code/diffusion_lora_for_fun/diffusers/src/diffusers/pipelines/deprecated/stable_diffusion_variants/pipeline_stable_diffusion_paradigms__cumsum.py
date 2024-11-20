def _cumsum(self, input, dim, debug=False):
    if debug:
        return torch.cumsum(input.cpu().float(), dim=dim).to(input.device)
    else:
        return torch.cumsum(input, dim=dim)
