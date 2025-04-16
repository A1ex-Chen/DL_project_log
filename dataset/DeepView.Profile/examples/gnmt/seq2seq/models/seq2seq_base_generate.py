def generate(self, inputs, context, beam_size):
    """
        Autoregressive generator, works with SequenceGenerator class.
        Executes decoder (in inference mode), applies log_softmax and topK for
        inference with beam search decoding.

        :param inputs: tensor with inputs to the decoder
        :param context: context from the encoder
        :param beam_size: beam size for the generator

        returns: (words, logprobs, scores, new_context)
            words: indices of topK tokens
            logprobs: log probabilities of topK tokens
            scores: scores from the attention module (for coverage penalty)
            new_context: new decoder context, includes new hidden states for
                decoder RNN cells
        """
    logits, scores, new_context = self.decode(inputs, context, True)
    logprobs = log_softmax(logits, dim=-1)
    logprobs, words = logprobs.topk(beam_size, dim=-1)
    return words, logprobs, scores, new_context
