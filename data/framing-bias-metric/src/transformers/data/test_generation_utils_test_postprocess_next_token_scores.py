def test_postprocess_next_token_scores(self):
    config = self.config
    model = self.model
    input_ids = torch.arange(0, 96, 1).view((8, 12))
    eos = config.eos_token_id
    bad_words_ids_test_cases = [[[299]], [[23, 24], [54]], [[config.
        eos_token_id]], []]
    masked_scores = [[(0, 299), (1, 299), (2, 299), (3, 299), (4, 299), (5,
        299), (6, 299), (7, 299)], [(1, 24), (0, 54), (1, 54), (2, 54), (3,
        54), (4, 54), (5, 54), (6, 54), (7, 54)], [(0, eos), (1, eos), (2,
        eos), (3, eos), (4, eos), (5, eos), (6, eos), (7, eos)], []]
    for test_case_index, bad_words_ids in enumerate(bad_words_ids_test_cases):
        scores = torch.rand((8, 300))
        output = model.postprocess_next_token_scores(scores, input_ids, 0,
            bad_words_ids, 13, 15, config.max_length, config.eos_token_id,
            config.repetition_penalty, 32, 5)
        for masked_score in masked_scores[test_case_index]:
            self.assertTrue(output[masked_score[0], masked_score[1]] == -
                float('inf'))
