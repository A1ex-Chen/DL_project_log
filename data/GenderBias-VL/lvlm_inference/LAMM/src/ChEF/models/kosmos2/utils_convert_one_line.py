def convert_one_line(idx, input_str):
    token = []
    img_src_token = []
    img_gpt_input_mask = []
    segments = input_str.split('<tab>')
    token.append(bos_id)
    img_gpt_input_mask.append(0)
    img_id = 0
    for i, segment in enumerate(segments):
        if segment.startswith('[image]'):
            if incontext_cfg is not None and incontext_cfg['use_pic']:
                image = images[idx * incontext_cfg['ice_num'] + img_id]
            else:
                image = images[idx]
            image_tensor = square_transform(self.args.input_resolution)(image)
            img_src_token.append(image_tensor)
            token.extend([boi_id] + list(range(4, image_feature_length + 4)
                ) + [eoi_id])
            img_gpt_input_mask.extend([0] + [1] * image_feature_length + [0])
            img_id += 1
        else:
            split_special_token_words = []
            split_results = split_string(segment, special_tokens)
            for string in split_results:
                if string in special_tokens:
                    split_special_token_words.append(string)
                else:
                    encode_tokens = tokenizer.encode(string, out_type=str)
                    split_special_token_words.extend(encode_tokens)
            segment = ' '.join(split_special_token_words)
            text_tokens = self.source_dictionary.encode_line(segment,
                add_if_not_exist=False).tolist()
            text_tokens = text_tokens[:-1]
            token.extend(text_tokens)
            img_gpt_input_mask.extend([0] * len(text_tokens))
    token.append(eos_id)
    assert len(token) == len(img_gpt_input_mask) + 1
    token = torch.LongTensor(token)
    img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)
    img_src_token = torch.stack(img_src_token, dim=0)
    return token, img_src_token, img_gpt_input_mask
