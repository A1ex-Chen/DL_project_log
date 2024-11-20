def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
    """Tie or clone module weights depending of whether we are using TorchScript or not"""
    if self.config.torchscript:
        output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone()
            )
    else:
        output_embeddings.weight = input_embeddings.weight
    if getattr(output_embeddings, 'bias', None) is not None:
        output_embeddings.bias.data = torch.nn.functional.pad(output_embeddings
            .bias.data, (0, output_embeddings.weight.shape[0] -
            output_embeddings.bias.shape[0]), 'constant', 0)
    if hasattr(output_embeddings, 'out_features') and hasattr(input_embeddings,
        'num_embeddings'):
        output_embeddings.out_features = input_embeddings.num_embeddings
