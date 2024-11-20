def tie_weights(self):
    """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
    output_embeddings = self.get_output_embeddings()
    if output_embeddings is not None:
        self._tie_or_clone_weights(output_embeddings, self.
            get_input_embeddings())
