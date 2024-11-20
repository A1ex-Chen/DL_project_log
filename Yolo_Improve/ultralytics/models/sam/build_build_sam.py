def build_sam(ckpt='sam_b.pt'):
    """Build a SAM model specified by ckpt."""
    model_builder = None
    ckpt = str(ckpt)
    for k in sam_model_map.keys():
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)
    if not model_builder:
        raise FileNotFoundError(
            f"""{ckpt} is not a supported SAM model. Available models are: 
 {sam_model_map.keys()}"""
            )
    return model_builder(ckpt)
