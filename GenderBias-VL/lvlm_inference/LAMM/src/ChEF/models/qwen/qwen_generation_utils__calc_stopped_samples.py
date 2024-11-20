def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) ->Iterable[int]:
    stopped_samples = []
    for prev_input_ids_slice in prev_input_ids:
        match = False
        for stop_token_seq in self.stop_words_ids:
            if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                match = True
                break
        stopped_samples.append(match)
    return stopped_samples
