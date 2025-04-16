def random_uniform(self, start, end):
    val = torch.rand(1).item()
    return start + (end - start) * val
