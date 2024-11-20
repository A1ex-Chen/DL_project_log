def evaluate(self, epoch, iteration, summary):
    """
        Runs evaluation on test dataset.

        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param summary: if True prints summary
        """
    batch_time = AverageMeter(False)
    tot_tok_per_sec = AverageMeter(False)
    iterations = AverageMeter(False)
    enc_seq_len = AverageMeter(False)
    dec_seq_len = AverageMeter(False)
    stats = {}
    output = []
    for i, (src, indices) in enumerate(self.loader):
        translate_timer = time.time()
        src, src_length = src
        batch_size = self.loader.batch_size
        global_batch_size = batch_size * get_world_size()
        beam_size = self.beam_size
        bos = [self.insert_target_start] * (batch_size * beam_size)
        bos = torch.LongTensor(bos)
        if self.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)
        src_length = torch.LongTensor(src_length)
        stats['total_enc_len'] = int(src_length.sum())
        if self.cuda:
            src = src.cuda()
            src_length = src_length.cuda()
            bos = bos.cuda()
        with torch.no_grad():
            context = self.model.encode(src, src_length)
            context = [context, src_length, None]
            if beam_size == 1:
                generator = self.generator.greedy_search
            else:
                generator = self.generator.beam_search
            preds, lengths, counter = generator(batch_size, bos, context)
        stats['total_dec_len'] = lengths.sum().item()
        stats['iters'] = counter
        indices = torch.tensor(indices).to(preds)
        preds = preds.scatter(0, indices.unsqueeze(1).expand_as(preds), preds)
        preds = gather_predictions(preds).cpu()
        for pred in preds:
            pred = pred.tolist()
            detok = self.tokenizer.detokenize(pred)
            output.append(detok + '\n')
        elapsed = time.time() - translate_timer
        batch_time.update(elapsed, batch_size)
        total_tokens = stats['total_dec_len'] + stats['total_enc_len']
        ttps = total_tokens / elapsed
        tot_tok_per_sec.update(ttps, batch_size)
        iterations.update(stats['iters'])
        enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
        dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)
        if i % self.print_freq == 0:
            log = []
            log += 'TEST '
            if epoch is not None:
                log += f'[{epoch}]'
            if iteration is not None:
                log += f'[{iteration}]'
            log += f'[{i}/{len(self.loader)}]\t'
            log += f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            log += (
                f'Decoder iters {iterations.val:.1f} ({iterations.avg:.1f})\t')
            log += (
                f'Tok/s {tot_tok_per_sec.val:.0f} ({tot_tok_per_sec.avg:.0f})')
            log = ''.join(log)
            logging.info(log)
    tot_tok_per_sec.reduce('sum')
    enc_seq_len.reduce('mean')
    dec_seq_len.reduce('mean')
    batch_time.reduce('mean')
    iterations.reduce('sum')
    if summary and get_rank() == 0:
        time_per_sentence = batch_time.avg / global_batch_size
        log = []
        log += 'TEST SUMMARY:\n'
        log += f'Lines translated: {len(self.loader.dataset)}\t'
        log += f'Avg total tokens/s: {tot_tok_per_sec.avg:.0f}\n'
        log += f'Avg time per batch: {batch_time.avg:.3f} s\t'
        log += f'Avg time per sentence: {1000 * time_per_sentence:.3f} ms\n'
        log += f'Avg encoder seq len: {enc_seq_len.avg:.2f}\t'
        log += f'Avg decoder seq len: {dec_seq_len.avg:.2f}\t'
        log += f'Total decoder iterations: {int(iterations.sum)}'
        log = ''.join(log)
        logging.info(log)
    return output
