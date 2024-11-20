def _maybe_expand_lora_scales_for_one_adapter(scales: Union[float, Dict],
    blocks_with_transformer: Dict[str, int], transformer_per_block: Dict[
    str, int], state_dict: None, default_scale: float=1.0):
    """
    Expands the inputs into a more granular dictionary. See the example below for more details.

    Parameters:
        scales (`Union[float, Dict]`):
            Scales dict to expand.
        blocks_with_transformer (`Dict[str, int]`):
            Dict with keys 'up' and 'down', showing which blocks have transformer layers
        transformer_per_block (`Dict[str, int]`):
            Dict with keys 'up' and 'down', showing how many transformer layers each block has

    E.g. turns
    ```python
    scales = {"down": 2, "mid": 3, "up": {"block_0": 4, "block_1": [5, 6, 7]}}
    blocks_with_transformer = {"down": [1, 2], "up": [0, 1]}
    transformer_per_block = {"down": 2, "up": 3}
    ```
    into
    ```python
    {
        "down.block_1.0": 2,
        "down.block_1.1": 2,
        "down.block_2.0": 2,
        "down.block_2.1": 2,
        "mid": 3,
        "up.block_0.0": 4,
        "up.block_0.1": 4,
        "up.block_0.2": 4,
        "up.block_1.0": 5,
        "up.block_1.1": 6,
        "up.block_1.2": 7,
    }
    ```
    """
    if sorted(blocks_with_transformer.keys()) != ['down', 'up']:
        raise ValueError(
            "blocks_with_transformer needs to be a dict with keys `'down' and `'up'`"
            )
    if sorted(transformer_per_block.keys()) != ['down', 'up']:
        raise ValueError(
            "transformer_per_block needs to be a dict with keys `'down' and `'up'`"
            )
    if not isinstance(scales, dict):
        return scales
    scales = copy.deepcopy(scales)
    if 'mid' not in scales:
        scales['mid'] = default_scale
    elif isinstance(scales['mid'], list):
        if len(scales['mid']) == 1:
            scales['mid'] = scales['mid'][0]
        else:
            raise ValueError(
                f"Expected 1 scales for mid, got {len(scales['mid'])}.")
    for updown in ['up', 'down']:
        if updown not in scales:
            scales[updown] = default_scale
        if not isinstance(scales[updown], dict):
            scales[updown] = {f'block_{i}': copy.deepcopy(scales[updown]) for
                i in blocks_with_transformer[updown]}
        for i in blocks_with_transformer[updown]:
            block = f'block_{i}'
            if block not in scales[updown]:
                scales[updown][block] = default_scale
            if not isinstance(scales[updown][block], list):
                scales[updown][block] = [scales[updown][block] for _ in
                    range(transformer_per_block[updown])]
            elif len(scales[updown][block]) == 1:
                scales[updown][block] = scales[updown][block
                    ] * transformer_per_block[updown]
            elif len(scales[updown][block]) != transformer_per_block[updown]:
                raise ValueError(
                    f'Expected {transformer_per_block[updown]} scales for {updown}.{block}, got {len(scales[updown][block])}.'
                    )
        for i in blocks_with_transformer[updown]:
            block = f'block_{i}'
            for tf_idx, value in enumerate(scales[updown][block]):
                scales[f'{updown}.{block}.{tf_idx}'] = value
        del scales[updown]
    for layer in scales.keys():
        if not any(_translate_into_actual_layer_name(layer) in module for
            module in state_dict.keys()):
            raise ValueError(
                f"Can't set lora scale for layer {layer}. It either doesn't exist in this unet or it has no attentions."
                )
    return {_translate_into_actual_layer_name(name): weight for name,
        weight in scales.items()}
