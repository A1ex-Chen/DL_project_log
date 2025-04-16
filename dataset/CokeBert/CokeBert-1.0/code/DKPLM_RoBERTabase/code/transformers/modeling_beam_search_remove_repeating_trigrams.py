def remove_repeating_trigrams(self, log_probabilities, _B):
    if self._step + 1 > 3:
        for i in range(_B * self.beam_size):
            tokens = [t for t in self.growing_beam[i]]
            trigrams = [(tokens[i - 1], tokens[i], tokens[i + 1]) for i in
                range(1, len(words) - 1)]
            last_trigram = tuple(trigrams[-1])
            if last_trigram in trigrams[:-1]:
                log_probabilities[i] = -1e+20
