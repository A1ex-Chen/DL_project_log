def span_to_answer(self, text: str, start: int, end: int) ->Dict[str, Union
    [str, int]]:
    """
        When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

        Args:
            text (:obj:`str`): The actual context to extract the answer from.
            start (:obj:`int`): The answer starting token index.
            end (:obj:`int`): The answer end token index.

        Returns:
            Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
        """
    words = []
    token_idx = char_start_idx = char_end_idx = chars_idx = 0
    for i, word in enumerate(text.split(' ')):
        token = self.tokenizer.tokenize(word)
        if start <= token_idx <= end:
            if token_idx == start:
                char_start_idx = chars_idx
            if token_idx == end:
                char_end_idx = chars_idx + len(word)
            words += [word]
        if token_idx > end:
            break
        token_idx += len(token)
        chars_idx += len(word) + 1
    return {'answer': ' '.join(words), 'start': max(0, char_start_idx),
        'end': min(len(text), char_end_idx)}
