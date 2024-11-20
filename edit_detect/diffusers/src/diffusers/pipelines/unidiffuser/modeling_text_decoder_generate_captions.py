@torch.no_grad()
def generate_captions(self, features, eos_token_id, device):
    """
        Generate captions given text embedding features. Returns list[L].

        Args:
            features (`torch.Tensor` of shape `(B, L, D)`):
                Text embedding features to generate captions from.
            eos_token_id (`int`):
                The token ID of the EOS token for the text decoder model.
            device:
                Device to perform text generation on.

        Returns:
            `List[str]`: A list of strings generated from the decoder model.
        """
    features = torch.split(features, 1, dim=0)
    generated_tokens = []
    generated_seq_lengths = []
    for feature in features:
        feature = self.decode_prefix(feature.to(device))
        output_tokens, seq_lengths = self.generate_beam(input_embeds=
            feature, device=device, eos_token_id=eos_token_id)
        generated_tokens.append(output_tokens[0])
        generated_seq_lengths.append(seq_lengths[0])
    generated_tokens = torch.stack(generated_tokens)
    generated_seq_lengths = torch.stack(generated_seq_lengths)
    return generated_tokens, generated_seq_lengths
