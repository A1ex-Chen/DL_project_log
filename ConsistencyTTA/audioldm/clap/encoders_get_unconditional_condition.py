def get_unconditional_condition(self, batchsize):
    self.unconditional_token = self.model.get_text_embedding(self.tokenizer
        (['', '']))[0:1]
    return torch.cat([self.unconditional_token.unsqueeze(0)] * batchsize, dim=0
        )
