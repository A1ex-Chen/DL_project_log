def convert_to_tensor(self, symbols):
    return torch.LongTensor(self.convert_tokens_to_ids(symbols))
