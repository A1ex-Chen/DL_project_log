@staticmethod
def _force_token_id_to_be_generated(scores, token_id) ->None:
    """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
    scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float(
        'inf')
