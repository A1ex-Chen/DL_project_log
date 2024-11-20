def generate_language_model(self, inputs_embeds: torch.Tensor=None,
    max_new_tokens: int=8, **model_kwargs):
    """

        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
    max_new_tokens = (max_new_tokens if max_new_tokens is not None else
        self.language_model.config.max_new_tokens)
    for _ in range(max_new_tokens):
        model_inputs = prepare_inputs_for_generation(inputs_embeds, **
            model_kwargs)
        output = self.language_model(**model_inputs, return_dict=True)
        next_hidden_states = output.last_hidden_state
        inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:,
            :]], dim=1)
        model_kwargs = self.language_model._update_model_kwargs_for_generation(
            output, model_kwargs)
    return inputs_embeds[:, -max_new_tokens:, :]
