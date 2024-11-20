def create_position_ids_from_inputs_embeds(self, inputs_embeds):
    """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]
    position_ids = torch.arange(self.padding_idx + 1, sequence_length +
        self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
    return position_ids.unsqueeze(0).expand(input_shape)
