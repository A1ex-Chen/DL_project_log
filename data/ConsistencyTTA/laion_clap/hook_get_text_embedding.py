def get_text_embedding(self, x, tokenizer=None, use_tensor=False):
    """get text embeddings from texts

        Parameters
        ----------
        x: List[str] (N,): 
            text list 
        tokenizer: func:
            the tokenizer function, if not provided (None), will use the default Roberta tokenizer.
        use_tensor: boolean:
            if True, the output will be the tesnor, preserving the gradient (default: False).      
        Returns
        ----------
        text_embed : numpy.darray | torch.Tensor (N,D):
            text embeddings that extracted from texts
        """
    self.model.eval()
    if tokenizer is not None:
        text_input = tokenizer(x)
    else:
        text_input = self.tokenizer(x)
    text_embed = self.model.get_text_embedding(text_input)
    if not use_tensor:
        text_embed = text_embed.detach().cpu().numpy()
    return text_embed
