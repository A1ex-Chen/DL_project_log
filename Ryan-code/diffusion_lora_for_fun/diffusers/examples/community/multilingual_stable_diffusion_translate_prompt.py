def translate_prompt(prompt, translation_tokenizer, translation_model, device):
    """helper function to translate prompt to English"""
    encoded_prompt = translation_tokenizer(prompt, return_tensors='pt').to(
        device)
    generated_tokens = translation_model.generate(**encoded_prompt,
        max_new_tokens=1000)
    en_trans = translation_tokenizer.batch_decode(generated_tokens,
        skip_special_tokens=True)
    return en_trans[0]
