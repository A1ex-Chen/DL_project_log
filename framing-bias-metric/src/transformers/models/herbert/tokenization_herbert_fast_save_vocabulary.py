def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    files = self._tokenizer.model.save(save_directory, name=filename_prefix)
    return tuple(files)
