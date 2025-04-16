def is_pretrained_params(n):
    return n.startswith('clap_model.transformer') or n in [
        'clap_model.positional_embedding', 'clap_model.text_projection'
        ] or n.startswith('clap_model.token_embedding') or n.startswith(
        'clap_model.ln_final') or n.startswith('clap_model.logit_scale_t')
