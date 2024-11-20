def random_mask(self, length):
    return torch.LongTensor(length).random_(0, 2)
