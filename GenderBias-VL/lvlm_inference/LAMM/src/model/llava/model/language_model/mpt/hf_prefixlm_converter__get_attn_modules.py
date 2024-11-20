def _get_attn_modules(model: CAUSAL_GPT_TYPES) ->List[torch.nn.Module]:
    """Helper that gets a list of the model's attention modules.

        Each module has a `bias` buffer used for causal masking. The Prefix LM
        conversion adds logic to dynamically manipulate these biases to support
        Prefix LM attention masking.
        """
    attn_modules = []
    if isinstance(model, GPTNeoXForCausalLM):
        blocks = model.gpt_neox.layers
    else:
        blocks = model.transformer.h
    for block in blocks:
        if isinstance(model, GPTNeoForCausalLM):
            if block.attn.attention_type != 'global':
                continue
            attn_module = block.attn.attention
        elif isinstance(model, GPTNeoXForCausalLM):
            attn_module = block.attention
        else:
            attn_module = block.attn
        attn_modules.append(attn_module)
    return attn_modules
