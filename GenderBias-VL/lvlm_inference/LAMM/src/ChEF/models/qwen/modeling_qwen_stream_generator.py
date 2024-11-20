def stream_generator():
    outputs = []
    for token in self.generate_stream(input_ids, return_dict_in_generate=
        False, generation_config=stream_config, logits_processor=
        logits_processor, seed=-1, **kwargs):
        outputs.append(token.item())
        yield tokenizer.decode(outputs, skip_special_tokens=True, errors=
            'ignore', keep_image_special=True)
