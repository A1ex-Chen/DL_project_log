def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    return self.gpt2_tokenizer.save_pretrained(save_directory,
        filename_prefix=filename_prefix)
