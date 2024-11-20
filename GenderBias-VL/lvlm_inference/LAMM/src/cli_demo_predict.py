def predict(args, model, input, images, image_path, pcl_path, chatbot,
    max_length, top_p, temperature, history, modality_cache, show_prompt=False
    ):
    if image_path is None and pcl_path is None and images is None:
        return [(input,
            'There is no input data provided! Please upload your data and start the conversation.'
            )]
    else:
        pass
    start = time.time()
    prompt_text = generate_conversation_text(args, input, history)
    if show_prompt:
        print(f'[!] prompt text: \n\t{prompt_text}', flush=True)
    if image_path:
        if isinstance(image_path, list):
            image_paths = image_path
        else:
            image_paths = [image_path]
    else:
        image_paths = []
    response = model.generate({'prompt': [prompt_text] if not isinstance(
        prompt_text, list) else prompt_text, 'image_paths': image_paths,
        'pcl_paths': [pcl_path] if pcl_path else [], 'images': [images] if
        images else [], 'top_p': top_p, 'temperature': temperature,
        'max_tgt_len': max_length, 'modality_embeds': modality_cache})
    history.append((input, response))
    return chatbot, history, modality_cache, time.time() - start
