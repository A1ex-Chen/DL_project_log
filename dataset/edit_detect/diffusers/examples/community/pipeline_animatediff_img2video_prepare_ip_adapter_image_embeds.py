def prepare_ip_adapter_image_embeds(self, ip_adapter_image,
    ip_adapter_image_embeds, device, num_images_per_prompt):
    if ip_adapter_image_embeds is None:
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]
        if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.
            image_projection_layers):
            raise ValueError(
                f'`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters.'
                )
        image_embeds = []
        for single_ip_adapter_image, image_proj_layer in zip(ip_adapter_image,
            self.unet.encoder_hid_proj.image_projection_layers):
            output_hidden_state = not isinstance(image_proj_layer,
                ImageProjection)
            single_image_embeds, single_negative_image_embeds = (self.
                encode_image(single_ip_adapter_image, device, 1,
                output_hidden_state))
            single_image_embeds = torch.stack([single_image_embeds] *
                num_images_per_prompt, dim=0)
            single_negative_image_embeds = torch.stack([
                single_negative_image_embeds] * num_images_per_prompt, dim=0)
            if self.do_classifier_free_guidance:
                single_image_embeds = torch.cat([
                    single_negative_image_embeds, single_image_embeds])
                single_image_embeds = single_image_embeds.to(device)
            image_embeds.append(single_image_embeds)
    else:
        image_embeds = ip_adapter_image_embeds
    return image_embeds
