def random_doc(self, length, num_vocab):
    return torch.LongTensor(length).random_(0, num_vocab + 1)
