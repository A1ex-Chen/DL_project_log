def tie_weights(self):
    """
        Tie the weights between the input embeddings and the output embeddings. If the `torchscript` flag is set in the
        configuration, can't handle parameter sharing so we are cloning the weights instead.
        """
    output_embeddings = self.get_output_embeddings()
    input_embeddings = self.get_input_embeddings()
    resized_dense = nn.Linear(input_embeddings.num_embeddings, self.config.
        hidden_size - self.config.embedding_size, bias=False)
    kept_data = self.cls.predictions.dense.weight.data[..., :min(self.cls.
        predictions.dense.weight.data.shape[1], resized_dense.weight.data.
        shape[1])]
    resized_dense.weight.data[..., :self.cls.predictions.dense.weight.data.
        shape[1]] = kept_data
    self.cls.predictions.dense = resized_dense
    self.cls.predictions.dense.to(self.device)
    if output_embeddings is not None and self.config.tie_word_embeddings:
        self._tie_or_clone_weights(output_embeddings, self.
            get_input_embeddings())
