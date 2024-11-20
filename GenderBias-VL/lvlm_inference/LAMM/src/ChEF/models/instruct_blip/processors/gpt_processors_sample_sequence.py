def sample_sequence(self, caption, history, answer):
    bos, eos, speaker1, speaker2, cap = self.tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-2])
    instance = {}
    sequence = [caption] + history + [answer]
    sequence = [(s + [eos]) for s in sequence]
    instance['input_ids'] = list(chain(*sequence))
    instance['token_type_ids'] = [cap] * len(sequence[0]) + [(speaker2 if i %
        2 else speaker1) for i, s in enumerate(sequence[1:]) for _ in s]
    instance['labels'] = [-1] * sum(len(s) for s in sequence[:-1]) + sequence[
        -1]
    assert len(instance['input_ids']) == len(instance['token_type_ids'])
    assert len(instance['token_type_ids']) == len(instance['labels'])
    for k, v in instance.items():
        instance[k] = torch.Tensor(v).long()
    return instance
