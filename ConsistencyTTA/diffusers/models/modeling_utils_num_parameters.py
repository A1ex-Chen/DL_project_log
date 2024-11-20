def num_parameters(self, only_trainable: bool=False, exclude_embeddings:
    bool=False) ->int:
    """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """
    if exclude_embeddings:
        embedding_param_names = [f'{name}.weight' for name, module_type in
            self.named_modules() if isinstance(module_type, torch.nn.Embedding)
            ]
        non_embedding_parameters = [parameter for name, parameter in self.
            named_parameters() if name not in embedding_param_names]
        return sum(p.numel() for p in non_embedding_parameters if p.
            requires_grad or not only_trainable)
    else:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or
            not only_trainable)
