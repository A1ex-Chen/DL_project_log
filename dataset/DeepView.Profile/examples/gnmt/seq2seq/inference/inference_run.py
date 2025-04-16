def run(self, calc_bleu=True, epoch=None, iteration=None, eval_path=None,
    summary=False, reference_path=None):
    """
        Runs translation on test dataset.

        :param calc_bleu: if True compares results with reference and computes
            BLEU score
        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param eval_path: path to the file for saving results
        :param summary: if True prints summary
        :param reference_path: path to the file with reference translation
        """
    if self.cuda:
        test_bleu = torch.cuda.FloatTensor([0])
        break_training = torch.cuda.LongTensor([0])
    else:
        test_bleu = torch.FloatTensor([0])
        break_training = torch.LongTensor([0])
    if eval_path is None:
        eval_path = self.build_eval_path(epoch, iteration)
    detok_eval_path = eval_path + '.detok'
    with contextlib.suppress(FileNotFoundError):
        os.remove(eval_path)
        os.remove(detok_eval_path)
    rank = get_rank()
    logging.info('Running evaluation on test set')
    self.model.eval()
    torch.cuda.empty_cache()
    output = self.evaluate(epoch, iteration, summary)
    output = output[:len(self.loader.dataset)]
    output = self.loader.dataset.unsort(output)
    if rank == 0:
        with open(eval_path, 'a') as eval_file:
            eval_file.writelines(output)
        if calc_bleu:
            self.run_detokenizer(eval_path)
            test_bleu[0] = self.run_sacrebleu(detok_eval_path, reference_path)
            if summary:
                logging.info(f'BLEU on test dataset: {test_bleu[0]:.2f}')
            if self.target_bleu and test_bleu[0] >= self.target_bleu:
                logging.info('Target accuracy reached')
                break_training[0] = 1
    barrier()
    torch.cuda.empty_cache()
    logging.info('Finished evaluation on test set')
    if self.distributed:
        dist.broadcast(break_training, 0)
        dist.broadcast(test_bleu, 0)
    return test_bleu[0].item(), break_training[0].item()
