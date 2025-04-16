def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
    self.u2seq = u2seq
    self.users = sorted(self.u2seq.keys())
    self.u2answer = u2answer
    self.max_len = max_len
    self.mask_token = mask_token
    self.negative_samples = negative_samples
