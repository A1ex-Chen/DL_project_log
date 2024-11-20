def construct_direction(self, embs_source: torch.Tensor, embs_target: torch
    .Tensor):
    """Constructs the edit direction to steer the image generation process semantically."""
    return (embs_target.mean(0) - embs_source.mean(0)).unsqueeze(0)
