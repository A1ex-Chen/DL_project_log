@staticmethod
def _compute_max_attention_per_index(attention_maps: torch.Tensor, indices:
    List[int]) ->List[torch.Tensor]:
    """Computes the maximum attention value for each of the tokens we wish to alter."""
    attention_for_text = attention_maps[:, :, 1:-1]
    attention_for_text *= 100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1
        )
    indices = [(index - 1) for index in indices]
    max_indices_list = []
    for i in indices:
        image = attention_for_text[:, :, i]
        smoothing = GaussianSmoothing().to(attention_maps.device)
        input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=
            'reflect')
        image = smoothing(input).squeeze(0).squeeze(0)
        max_indices_list.append(image.max())
    return max_indices_list
