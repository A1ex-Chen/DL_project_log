def assign_to_checkpoint(paths, checkpoint, old_checkpoint,
    additional_replacements=None, config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list
        ), "Paths should be a list of dicts containing 'old' and 'new' keys."
    for path in paths:
        new_path = path['new']
        new_path = new_path.replace('middle_block.0', 'mid_block.resnets.0')
        new_path = new_path.replace('middle_block.1', 'mid_block.attentions.0')
        new_path = new_path.replace('middle_block.2', 'mid_block.resnets.1')
        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'], replacement
                    ['new'])
        if 'proj_attn.weight' in new_path or 'to_out.0.weight' in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path['old']]
