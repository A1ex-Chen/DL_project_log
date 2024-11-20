def decode_reference(self, model, x, out_lens):
    """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
    model = getattr(model, 'module', model)
    with torch.no_grad():
        logits, out_lens = model.encode(x, out_lens)
        output = []
        for batch_idx in range(logits.size(0)):
            inseq = logits[batch_idx, :, :].unsqueeze(1)
            logitlen = out_lens[batch_idx]
            sentence = self._greedy_decode(model, inseq, logitlen)
            output.append(sentence)
    return output
