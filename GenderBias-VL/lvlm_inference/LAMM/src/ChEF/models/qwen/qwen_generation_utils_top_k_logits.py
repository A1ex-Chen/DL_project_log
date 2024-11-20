def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None
            ]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True,
            dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
            dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1
            ].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
    return logits
