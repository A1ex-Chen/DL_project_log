def embeddings_from_original_checkpoint(model, diffuser_embeddings_prefix,
    original_embeddings_prefix):
    embeddings = {}
    embeddings.update({
        f'{diffuser_embeddings_prefix}.word_embeddings.weight': model[
        f'{original_embeddings_prefix}.word_embeddings.weight']})
    embeddings.update({
        f'{diffuser_embeddings_prefix}.position_embeddings.weight': model[
        f'{original_embeddings_prefix}.position_embeddings.weight']})
    embeddings.update({f'{diffuser_embeddings_prefix}.LayerNorm.weight':
        model[f'{original_embeddings_prefix}.LayerNorm.weight']})
    embeddings.update({f'{diffuser_embeddings_prefix}.LayerNorm.bias':
        model[f'{original_embeddings_prefix}.LayerNorm.bias']})
    return embeddings
