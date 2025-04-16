def preprocess_train(examples):
    preprocessed_images = preprocess_images(examples)
    original_images, edited_images = preprocessed_images.chunk(2)
    original_images = original_images.reshape(-1, 3, args.resolution, args.
        resolution)
    edited_images = edited_images.reshape(-1, 3, args.resolution, args.
        resolution)
    examples['original_pixel_values'] = original_images
    examples['edited_pixel_values'] = edited_images
    captions = list(examples[edit_prompt_column])
    prompt_embeds_all, add_text_embeds_all = compute_embeddings_for_prompts(
        captions, text_encoders, tokenizers)
    examples['prompt_embeds'] = prompt_embeds_all
    examples['add_text_embeds'] = add_text_embeds_all
    return examples
