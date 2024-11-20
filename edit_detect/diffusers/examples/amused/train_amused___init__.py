def __init__(self, hf_dataset, tokenizer, image_key, prompt_key,
    prompt_prefix=None, size=512):
    self.size = size
    self.image_key = image_key
    self.prompt_key = prompt_key
    self.tokenizer = tokenizer
    self.hf_dataset = hf_dataset
    self.prompt_prefix = prompt_prefix
