def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids,
    attention_mask, past_key_values, labels, images):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if (past_key_values is not None and vision_tower is not None and 
            images is not None and input_ids.shape[1] == 1):
            target_shape = past_key_values[-1][-1].shape[-2] + 1
            attention_mask = torch.cat((attention_mask, torch.ones((
                attention_mask.shape[0], target_shape - attention_mask.
                shape[1]), dtype=attention_mask.dtype, device=
                attention_mask.device)), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        return (input_ids, position_ids, attention_mask, past_key_values,
            None, labels)
    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1).to(self.device) for x in
            image_features]
    else:
        image_features = self.encode_images(images).to(self.device)
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.
        config, 'mm_use_im_start_end', False):
        raise NotImplementedError
    print(f'image_features: {image_features.shape}')
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long,
            device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids,
        cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels,
        cur_attention_mask in zip(labels, attention_mask)]
    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1,
                cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        image_token_indices = [-1] + torch.where(cur_input_ids ==
            IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:
                image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(
            cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes,
            dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            print('cur_new_input_embeds_1', len(cur_new_input_embeds))
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                print('cur_new_input_embeds_2', len(cur_new_input_embeds))
                cur_new_input_embeds.append(cur_image_features)
                print('cur_new_input_embeds_3', len(cur_new_input_embeds))
                cur_new_labels.append(torch.full((cur_image_features.shape[
                    0],), IGNORE_INDEX, device=cur_labels.device, dtype=
                    cur_labels.dtype))
        print('0: ', cur_new_input_embeds[0].shape)
        print('image: ', cur_new_input_embeds[1].shape)
        print('2: ', cur_new_input_embeds[2].shape)
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
    tokenizer_model_max_length = getattr(self.config,
        'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in
            new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)
    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX,
        dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=
        attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.
        dtype, device=position_ids.device)
    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(
        new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == 'left':
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len -
                cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype
                    =position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.
                zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=
                cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=
                    position_ids.dtype, device=position_ids.device)
    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded
    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
    if _position_ids is None:
        position_ids = None
    return (None, position_ids, attention_mask, past_key_values,
        new_input_embeds, new_labels)
