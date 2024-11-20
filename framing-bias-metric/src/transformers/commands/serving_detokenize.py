def detokenize(self, tokens_ids: List[int]=Body(None, embed=True),
    skip_special_tokens: bool=Body(False, embed=True),
    cleanup_tokenization_spaces: bool=Body(True, embed=True)):
    """
        Detokenize the provided tokens ids to readable text: - **tokens_ids**: List of tokens ids -
        **skip_special_tokens**: Flag indicating to not try to decode special tokens - **cleanup_tokenization_spaces**:
        Flag indicating to remove all leading/trailing spaces and intermediate ones.
        """
    try:
        decoded_str = self._pipeline.tokenizer.decode(tokens_ids,
            skip_special_tokens, cleanup_tokenization_spaces)
        return ServeDeTokenizeResult(model='', text=decoded_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail={'model': '', 'error':
            str(e)})
