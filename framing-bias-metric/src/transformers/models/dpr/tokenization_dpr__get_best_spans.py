def _get_best_spans(self, start_logits: List[int], end_logits: List[int],
    max_answer_length: int, top_spans: int) ->List[DPRSpanPrediction]:
    """
        Finds the best answer span for the extractive Q&A model for one passage. It returns the best span by descending
        `span_score` order and keeping max `top_spans` spans. Spans longer that `max_answer_length` are ignored.
        """
    scores = []
    for start_index, start_score in enumerate(start_logits):
        for answer_length, end_score in enumerate(end_logits[start_index:
            start_index + max_answer_length]):
            scores.append(((start_index, start_index + answer_length), 
                start_score + end_score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    chosen_span_intervals = []
    for (start_index, end_index), score in scores:
        assert start_index <= end_index, 'Wrong span indices: [{}:{}]'.format(
            start_index, end_index)
        length = end_index - start_index + 1
        assert length <= max_answer_length, 'Span is too long: {} > {}'.format(
            length, max_answer_length)
        if any([(start_index <= prev_start_index <= prev_end_index <=
            end_index or prev_start_index <= start_index <= end_index <=
            prev_end_index) for prev_start_index, prev_end_index in
            chosen_span_intervals]):
            continue
        chosen_span_intervals.append((start_index, end_index))
        if len(chosen_span_intervals) == top_spans:
            break
    return chosen_span_intervals
