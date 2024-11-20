def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss."""
    return F.binary_cross_entropy(output_dict['clipwise_output'],
        target_dict['target'])
