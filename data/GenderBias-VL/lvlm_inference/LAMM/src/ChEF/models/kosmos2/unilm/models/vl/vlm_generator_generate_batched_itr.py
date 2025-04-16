def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None
    ):
    """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
    for sample in data_itr:
        s = utils.move_to_cuda(sample) if cuda else sample
        if 'net_input' not in s:
            continue
        input = s['net_input']
        encoder_input = {k: v for k, v in input.items() if k !=
            'prev_output_tokens'}
        if timer is not None:
            timer.start()
        with torch.no_grad():
            hypos = self.generate(encoder_input)
        if timer is not None:
            timer.stop(sum(len(h[0]['tokens']) for h in hypos))
        for i, id in enumerate(s['id'].data):
            src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
            ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s[
                'target'] is not None else None
            yield id, src, ref, hypos[i]
