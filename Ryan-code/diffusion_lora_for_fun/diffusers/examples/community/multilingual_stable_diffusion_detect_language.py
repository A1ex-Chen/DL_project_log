def detect_language(pipe, prompt, batch_size):
    """helper function to detect language(s) of prompt"""
    if batch_size == 1:
        preds = pipe(prompt, top_k=1, truncation=True, max_length=128)
        return preds[0]['label']
    else:
        detected_languages = []
        for p in prompt:
            preds = pipe(p, top_k=1, truncation=True, max_length=128)
            detected_languages.append(preds[0]['label'])
        return detected_languages
