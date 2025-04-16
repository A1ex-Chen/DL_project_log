@timeout_decorator.timeout(10)
def test_postprocess_next_token_scores_large_bad_words_list(self):
    config = self.config
    model = self.model
    input_ids = torch.arange(0, 96, 1).view((8, 12))
    bad_words_ids = []
    for _ in range(100):
        length_bad_word = random.randint(1, 4)
        bad_words_ids.append(random.sample(range(1, 300), length_bad_word))
    scores = torch.rand((8, 300))
    _ = model.postprocess_next_token_scores(scores, input_ids, 0,
        bad_words_ids, 13, 15, config.max_length, config.eos_token_id,
        config.repetition_penalty, 32, 5)
