def _get_logits_warper(self, top_k: int=None, top_p: float=None,
    temperature: float=None, num_beams: int=None) ->LogitsProcessorList:
    """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
        """
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    temperature = (temperature if temperature is not None else self.config.
        temperature)
    warpers = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=2 if
            num_beams > 1 else 1))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=2 if
            num_beams > 1 else 1))
    return warpers
