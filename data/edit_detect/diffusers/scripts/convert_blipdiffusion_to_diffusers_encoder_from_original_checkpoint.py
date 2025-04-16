def encoder_from_original_checkpoint(model, diffuser_encoder_prefix,
    original_encoder_prefix):
    encoder = {}
    for i in range(blip2config.qformer_config.num_hidden_layers):
        encoder.update(attention_from_original_checkpoint(model,
            f'{diffuser_encoder_prefix}.{i}.attention',
            f'{original_encoder_prefix}.{i}.attention'))
        encoder.update(attention_from_original_checkpoint(model,
            f'{diffuser_encoder_prefix}.{i}.crossattention',
            f'{original_encoder_prefix}.{i}.crossattention'))
        encoder.update({
            f'{diffuser_encoder_prefix}.{i}.intermediate.dense.weight':
            model[f'{original_encoder_prefix}.{i}.intermediate.dense.weight']})
        encoder.update({
            f'{diffuser_encoder_prefix}.{i}.intermediate.dense.bias': model
            [f'{original_encoder_prefix}.{i}.intermediate.dense.bias']})
        encoder.update({
            f'{diffuser_encoder_prefix}.{i}.intermediate_query.dense.weight':
            model[
            f'{original_encoder_prefix}.{i}.intermediate_query.dense.weight']})
        encoder.update({
            f'{diffuser_encoder_prefix}.{i}.intermediate_query.dense.bias':
            model[
            f'{original_encoder_prefix}.{i}.intermediate_query.dense.bias']})
        encoder.update(output_layers_from_original_checkpoint(model,
            f'{diffuser_encoder_prefix}.{i}.output',
            f'{original_encoder_prefix}.{i}.output'))
        encoder.update(output_layers_from_original_checkpoint(model,
            f'{diffuser_encoder_prefix}.{i}.output_query',
            f'{original_encoder_prefix}.{i}.output_query'))
    return encoder
