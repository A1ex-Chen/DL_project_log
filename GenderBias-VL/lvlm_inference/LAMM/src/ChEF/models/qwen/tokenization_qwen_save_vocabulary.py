def save_vocabulary(self, save_directory: str, **kwargs) ->Tuple[str]:
    """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
    file_path = os.path.join(save_directory, 'qwen.tiktoken')
    with open(file_path, 'w', encoding='utf8') as w:
        for k, v in self.mergeable_ranks.items():
            line = base64.b64encode(k).decode('utf8') + ' ' + str(v) + '\n'
            w.write(line)
    return file_path,
