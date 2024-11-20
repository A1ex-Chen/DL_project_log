def save_pretrained(self, path: str, filename_prefix: str=None):
    import torch
    filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
    if filename_prefix is not None:
        filename = filename_prefix + '-' + filename
    full_path = os.path.join(path, filename)
    torch.save(self.gpt2_encoder, full_path)
    return full_path,
