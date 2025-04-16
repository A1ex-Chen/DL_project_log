def create_ldm_bert_config(original_config):
    bert_params = original_config.model.parms.cond_stage_config.params
    config = LDMBertConfig(d_model=bert_params.n_embed, encoder_layers=
        bert_params.n_layer, encoder_ffn_dim=bert_params.n_embed * 4)
    return config
