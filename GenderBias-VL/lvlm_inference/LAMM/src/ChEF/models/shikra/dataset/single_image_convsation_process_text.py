def process_text(self, conv: Conversation) ->Dict[str, Any]:
    """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
    return self.process_func['text'](conv, self.preprocessor, self.mode, **
        self.tokenize_kwargs)
