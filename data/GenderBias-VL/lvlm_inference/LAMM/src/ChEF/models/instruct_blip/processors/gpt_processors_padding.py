def padding(self, seq):
    padded_seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True,
        padding_value=1.0)
    return padded_seq
