def _import_flash_attn():
    global apply_rotary_emb_func, rms_norm, flash_attn_unpadded_func, flash_attn_func
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func as __apply_rotary_emb_func
        apply_rotary_emb_func = __apply_rotary_emb_func
    except ImportError:
        logger.warn(
            'Warning: import flash_attn rotary fail, please install FlashAttention rotary to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary'
            )
    try:
        from flash_attn.ops.rms_norm import rms_norm as __rms_norm
        rms_norm = __rms_norm
    except ImportError:
        logger.warn(
            'Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm'
            )
    try:
        import flash_attn
        _flash_attn_func = None
        if not hasattr(flash_attn, '__version__'):
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        elif int(flash_attn.__version__.split('.')[0]) >= 2:
            if int(flash_attn.__version__.split('.')[1]) >= 1:
                from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as __flash_attn_unpadded_func
        else:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        flash_attn_unpadded_func = __flash_attn_unpadded_func
        flash_attn_func = _flash_attn_func
    except ImportError:
        logger.warn(
            'Warning: import flash_attn fail, please install FlashAttention to get higher efficiency https://github.com/Dao-AILab/flash-attention'
            )
