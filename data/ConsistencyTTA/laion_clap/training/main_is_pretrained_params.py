def is_pretrained_params(n):
    return n.startswith('transformer') or n in ['positional_embedding',
        'text_projection'] or n.startswith('token_embedding') or n.startswith(
        'ln_final') or n.startswith('logit_scale_t')
