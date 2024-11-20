def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool
    ) ->torch.Tensor:
    """Embeds point prompts."""
    points = points + 0.5
    if pad:
        padding_point = torch.zeros((points.shape[0], 1, 2), device=points.
            device)
        padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
    point_embedding = self.pe_layer.forward_with_coords(points, self.
        input_image_size)
    point_embedding[labels == -1] = 0.0
    point_embedding[labels == -1] += self.not_a_point_embed.weight
    point_embedding[labels == 0] += self.point_embeddings[0].weight
    point_embedding[labels == 1] += self.point_embeddings[1].weight
    return point_embedding
