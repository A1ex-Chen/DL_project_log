def get_tokenize_len(prompts):
    return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts
        ]
