def transformer_layer_load_qkvo(transformerLayerWeight:
    TransformerLayerWeight, lora_weights, lora_config: LoraConfig):
    lora_scaling_ = lora_config.lora_alpha / lora_config.r
    n_embed = transformerLayerWeight.network_config_['hidden_size']
    split_n_embed = n_embed // transformerLayerWeight.world_size_
    q_lora_A_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.q_proj.lora_A.default.weight'
        ]
    q_lora_B_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.q_proj.lora_B.default.weight'
        ]
    q_lora_weight_ = torch.mm(q_lora_B_weight_.cuda(), q_lora_A_weight_.cuda()
        ) * lora_scaling_
    q_lora_weight_ = q_lora_weight_[split_n_embed * transformerLayerWeight.
        tp_rank_:split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    q_lora_weight_ = q_lora_weight_.transpose(0, 1).contiguous().to(
        transformerLayerWeight.data_type_)
    transformerLayerWeight.q_weight_ += q_lora_weight_
    k_lora_A_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.k_proj.lora_A.default.weight'
        ]
    k_lora_B_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.k_proj.lora_B.default.weight'
        ]
    k_lora_weight_ = torch.mm(k_lora_B_weight_.cuda(), k_lora_A_weight_.cuda()
        ) * lora_scaling_
    k_lora_weight_ = k_lora_weight_[split_n_embed * transformerLayerWeight.
        tp_rank_:split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    k_lora_weight_ = k_lora_weight_.transpose(0, 1).contiguous().to(
        transformerLayerWeight.data_type_)
    transformerLayerWeight.k_weight_ += k_lora_weight_
    v_lora_A_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.v_proj.lora_A.default.weight'
        ]
    v_lora_B_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.v_proj.lora_B.default.weight'
        ]
    v_lora_weight_ = torch.mm(v_lora_B_weight_.cuda(), v_lora_A_weight_.cuda()
        ) * lora_scaling_
    v_lora_weight_ = v_lora_weight_[split_n_embed * transformerLayerWeight.
        tp_rank_:split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    v_lora_weight_ = v_lora_weight_.transpose(0, 1).contiguous().to(
        transformerLayerWeight.data_type_)
    transformerLayerWeight.v_weight_ += v_lora_weight_
    o_lora_A_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.o_proj.lora_A.default.weight'
        ]
    o_lora_B_weight_ = lora_weights[
        f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.o_proj.lora_B.default.weight'
        ]
    o_lora_weight_ = torch.mm(o_lora_B_weight_.cuda(), o_lora_A_weight_.cuda()
        ) * lora_scaling_
    o_lora_weight_ = o_lora_weight_[split_n_embed * transformerLayerWeight.
        tp_rank_:split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    o_lora_weight_ = o_lora_weight_.transpose(0, 1).contiguous().to(
        transformerLayerWeight.data_type_)
    transformerLayerWeight.o_weight_ += o_lora_weight_
