def _clean_padding_history(self, generated_tensor) ->List[List[int]]:
    """
        Cleans the padding history. Padding may be generated in two places when multiple conversations are provided as
        an input:

            - at the end of the concatenated history and new user input, so that all input to the model have the same
              length
            - at the end of the generated response, as some responses will be longer than others
        This method cleans up these padding token so that the history for each conversation is not impacted by the
        batching process.
        """
    outputs = []
    for sequence in generated_tensor:
        sequence_tokens = []
        is_previous_pad = False
        for token in sequence:
            if token == self.tokenizer.pad_token_id:
                if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                    continue
                if is_previous_pad:
                    continue
                else:
                    is_previous_pad = True
            else:
                is_previous_pad = False
            if self.framework == 'pt':
                sequence_tokens.append(token.item())
            else:
                sequence_tokens.append(int(token.numpy()))
        outputs.append(sequence_tokens)
    return outputs
