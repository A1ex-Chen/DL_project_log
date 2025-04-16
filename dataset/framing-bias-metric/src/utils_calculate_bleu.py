def calculate_bleu(output_lns, refs_lns, **kwargs) ->dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {'bleu': round(corpus_bleu(output_lns, [refs_lns], **kwargs).
        score, 4)}
