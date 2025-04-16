def __call__(self, input):
    ar = np.array(input)
    return torch.from_numpy(ar).long()
