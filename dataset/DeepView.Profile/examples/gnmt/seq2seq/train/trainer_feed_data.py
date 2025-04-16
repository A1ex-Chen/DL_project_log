def feed_data(self, data_loader, training=True):
    """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """
    if training:
        assert self.optimizer is not None
        eval_fractions = np.linspace(0, 1, self.intra_epoch_eval + 2)[1:-1]
        iters_with_update = len(data_loader) // self.iter_size
        eval_iters = (eval_fractions * iters_with_update).astype(int)
        eval_iters = eval_iters * self.iter_size
        eval_iters = set(eval_iters)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_per_token = AverageMeter(skip_first=False)
    losses_per_sentence = AverageMeter(skip_first=False)
    tot_tok_time = AverageMeter()
    src_tok_time = AverageMeter()
    tgt_tok_time = AverageMeter()
    batch_size = data_loader.batch_size
    end = time.time()
    for i, (src, tgt) in enumerate(data_loader):
        self.save_counter += 1
        data_time.update(time.time() - end)
        update = False
        if i % self.iter_size == self.iter_size - 1:
            update = True
        stats = self.iterate(src, tgt, update, training=training)
        loss_per_token, loss_per_sentence, num_toks = stats
        losses_per_token.update(loss_per_token, num_toks['tgt'])
        losses_per_sentence.update(loss_per_sentence, batch_size)
        elapsed = time.time() - end
        batch_time.update(elapsed)
        src_tok_time.update(num_toks['src'] / elapsed)
        tgt_tok_time.update(num_toks['tgt'] / elapsed)
        tot_num_toks = num_toks['tgt'] + num_toks['src']
        tot_tok_time.update(tot_num_toks / elapsed)
        self.loss = losses_per_token.avg
        if training and i in eval_iters:
            test_bleu, _ = self.translator.run(calc_bleu=True, epoch=self.
                epoch, iteration=i)
            log = []
            log += [f'TRAIN [{self.epoch}][{i}/{len(data_loader)}]']
            log += [f'BLEU: {test_bleu:.2f}']
            log = '\t'.join(log)
            logging.info(log)
            self.model.train()
            self.preallocate(data_loader, training=True)
        if i % self.print_freq == 0:
            phase = 'TRAIN' if training else 'VALIDATION'
            log = []
            log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
            log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
            log += [f'Data {data_time.val:.2e} ({data_time.avg:.2e})']
            log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
            if self.verbose:
                log += [
                    f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})'
                    ]
                log += [
                    f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})'
                    ]
                log += [
                    f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})'
                    ]
            log += [
                f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})'
                ]
            if training:
                lr = self.optimizer.param_groups[0]['lr']
                log += [f'LR {lr:.3e}']
            log = '\t'.join(log)
            logging.info(log)
        save_chkpt = self.save_counter % self.save_freq == self.save_freq - 1
        if training and save_chkpt:
            self.save_counter = 0
            self.save_info['iteration'] = i
            identifier = next(self.checkpoint_counter, -1)
            if identifier != -1:
                with sync_workers() as rank:
                    if rank == 0:
                        self.save(identifier=identifier)
        end = time.time()
    tot_tok_time.reduce('sum')
    losses_per_token.reduce('mean')
    return losses_per_token.avg, tot_tok_time.avg
