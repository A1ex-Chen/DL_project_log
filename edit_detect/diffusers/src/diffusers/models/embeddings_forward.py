def forward(self, image_embeds: List[torch.Tensor]):
    projected_image_embeds = []
    if not isinstance(image_embeds, list):
        deprecation_message = (
            'You have passed a tensor as `image_embeds`.This is deprecated and will be removed in a future release. Please make sure to update your script to pass `image_embeds` as a list of tensors to supress this warning.'
            )
        deprecate('image_embeds not a list', '1.0.0', deprecation_message,
            standard_warn=False)
        image_embeds = [image_embeds.unsqueeze(1)]
    if len(image_embeds) != len(self.image_projection_layers):
        raise ValueError(
            f'image_embeds must have the same length as image_projection_layers, got {len(image_embeds)} and {len(self.image_projection_layers)}'
            )
    for image_embed, image_projection_layer in zip(image_embeds, self.
        image_projection_layers):
        batch_size, num_images = image_embed.shape[0], image_embed.shape[1]
        image_embed = image_embed.reshape((batch_size * num_images,) +
            image_embed.shape[2:])
        image_embed = image_projection_layer(image_embed)
        image_embed = image_embed.reshape((batch_size, num_images) +
            image_embed.shape[1:])
        projected_image_embeds.append(image_embed)
    return projected_image_embeds
