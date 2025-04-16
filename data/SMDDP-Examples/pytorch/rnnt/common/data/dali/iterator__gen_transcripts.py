def _gen_transcripts(self, labels, normalize_transcripts: bool=True):
    """
        Generate transcripts in format expected by NN
        """
    ids = labels.flatten().numpy()
    transcripts = [torch.tensor(self.tr[i]) for i in ids
        ] if self.jit_tensor_formation else [self.tr[i] for i in ids]
    transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True
        )
    return transcripts.cuda(), self.t_sizes[ids].cuda()
