def iterate(self, src, tgt, update=True, training=True):
    """
        Performs one iteration of the training/validation.

        :param src: batch of examples from the source language
        :param tgt: batch of examples from the target language
        :param update: if True: optimizer does update of the weights
        :param training: if True: executes optimizer
        """
    src, src_length = src
    tgt, tgt_length = tgt
    src_length = torch.LongTensor(src_length)
    tgt_length = torch.LongTensor(tgt_length)
    num_toks = {}
    num_toks['tgt'] = int(sum(tgt_length - 1))
    num_toks['src'] = int(sum(src_length))
    if self.cuda:
        src = src.cuda()
        src_length = src_length.cuda()
        tgt = tgt.cuda()
    if self.batch_first:
        output = self.model(src, src_length, tgt[:, :-1])
        tgt_labels = tgt[:, 1:]
        T, B = output.size(1), output.size(0)
    else:
        output = self.model(src, src_length, tgt[:-1])
        tgt_labels = tgt[1:]
        T, B = output.size(0), output.size(1)
    loss = self.criterion(output.view(T * B, -1), tgt_labels.contiguous().
        view(-1))
    loss_per_batch = loss.item()
    loss /= B * self.iter_size
    if training:
        self.fp_optimizer.step(loss, self.optimizer, self.scheduler, update)
    loss_per_token = loss_per_batch / num_toks['tgt']
    loss_per_sentence = loss_per_batch / B
    return loss_per_token, loss_per_sentence, num_toks
