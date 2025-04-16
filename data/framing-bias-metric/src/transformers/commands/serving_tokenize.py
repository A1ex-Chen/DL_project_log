def tokenize(self, text_input: str=Body(None, embed=True), return_ids: bool
    =Body(False, embed=True)):
    """
        Tokenize the provided input and eventually returns corresponding tokens id: - **text_input**: String to
        tokenize - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer
        mapping.
        """
    try:
        tokens_txt = self._pipeline.tokenizer.tokenize(text_input)
        if return_ids:
            tokens_ids = self._pipeline.tokenizer.convert_tokens_to_ids(
                tokens_txt)
            return ServeTokenizeResult(tokens=tokens_txt, tokens_ids=tokens_ids
                )
        else:
            return ServeTokenizeResult(tokens=tokens_txt)
    except Exception as e:
        raise HTTPException(status_code=500, detail={'model': '', 'error':
            str(e)})
