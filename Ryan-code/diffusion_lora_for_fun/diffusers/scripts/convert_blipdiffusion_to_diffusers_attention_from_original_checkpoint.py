def attention_from_original_checkpoint(model, diffuser_attention_prefix,
    original_attention_prefix):
    attention = {}
    attention.update({f'{diffuser_attention_prefix}.attention.query.weight':
        model[f'{original_attention_prefix}.self.query.weight']})
    attention.update({f'{diffuser_attention_prefix}.attention.query.bias':
        model[f'{original_attention_prefix}.self.query.bias']})
    attention.update({f'{diffuser_attention_prefix}.attention.key.weight':
        model[f'{original_attention_prefix}.self.key.weight']})
    attention.update({f'{diffuser_attention_prefix}.attention.key.bias':
        model[f'{original_attention_prefix}.self.key.bias']})
    attention.update({f'{diffuser_attention_prefix}.attention.value.weight':
        model[f'{original_attention_prefix}.self.value.weight']})
    attention.update({f'{diffuser_attention_prefix}.attention.value.bias':
        model[f'{original_attention_prefix}.self.value.bias']})
    attention.update({f'{diffuser_attention_prefix}.output.dense.weight':
        model[f'{original_attention_prefix}.output.dense.weight']})
    attention.update({f'{diffuser_attention_prefix}.output.dense.bias':
        model[f'{original_attention_prefix}.output.dense.bias']})
    attention.update({
        f'{diffuser_attention_prefix}.output.LayerNorm.weight': model[
        f'{original_attention_prefix}.output.LayerNorm.weight']})
    attention.update({f'{diffuser_attention_prefix}.output.LayerNorm.bias':
        model[f'{original_attention_prefix}.output.LayerNorm.bias']})
    return attention
