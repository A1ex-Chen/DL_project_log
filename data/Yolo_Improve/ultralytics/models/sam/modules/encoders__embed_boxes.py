def _embed_boxes(self, boxes: torch.Tensor) ->torch.Tensor:
    """Embeds box prompts."""
    boxes = boxes + 0.5
    coords = boxes.reshape(-1, 2, 2)
    corner_embedding = self.pe_layer.forward_with_coords(coords, self.
        input_image_size)
    corner_embedding[:, 0, :] += self.point_embeddings[2].weight
    corner_embedding[:, 1, :] += self.point_embeddings[3].weight
    return corner_embedding
