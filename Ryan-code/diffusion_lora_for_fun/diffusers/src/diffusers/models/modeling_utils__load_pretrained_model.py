@classmethod
def _load_pretrained_model(cls, model, state_dict: OrderedDict,
    resolved_archive_file, pretrained_model_name_or_path: Union[str, os.
    PathLike], ignore_mismatched_sizes: bool=False):
    model_state_dict = model.state_dict()
    loaded_keys = list(state_dict.keys())
    expected_keys = list(model_state_dict.keys())
    original_loaded_keys = loaded_keys
    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))
    model_to_load = model

    def _find_mismatched_keys(state_dict, model_state_dict, loaded_keys,
        ignore_mismatched_sizes):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key
                if model_key in model_state_dict and state_dict[checkpoint_key
                    ].shape != model_state_dict[model_key].shape:
                    mismatched_keys.append((checkpoint_key, state_dict[
                        checkpoint_key].shape, model_state_dict[model_key].
                        shape))
                    del state_dict[checkpoint_key]
        return mismatched_keys
    if state_dict is not None:
        mismatched_keys = _find_mismatched_keys(state_dict,
            model_state_dict, original_loaded_keys, ignore_mismatched_sizes)
        error_msgs = _load_state_dict_into_model(model_to_load, state_dict)
    if len(error_msgs) > 0:
        error_msg = '\n\t'.join(error_msgs)
        if 'size mismatch' in error_msg:
            error_msg += """
	You may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."""
        raise RuntimeError(
            f"""Error(s) in loading state_dict for {model.__class__.__name__}:
	{error_msg}"""
            )
    if len(unexpected_keys) > 0:
        logger.warning(
            f"""Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}
- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."""
            )
    else:
        logger.info(
            f"""All model checkpoint weights were used when initializing {model.__class__.__name__}.
"""
            )
    if len(missing_keys) > 0:
        logger.warning(
            f"""Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    elif len(mismatched_keys) == 0:
        logger.info(
            f"""All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.
If your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training."""
            )
    if len(mismatched_keys) > 0:
        mismatched_warning = '\n'.join([
            f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated'
             for key, shape1, shape2 in mismatched_keys])
        logger.warning(
            f"""Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized because the shapes did not match:
{mismatched_warning}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs
