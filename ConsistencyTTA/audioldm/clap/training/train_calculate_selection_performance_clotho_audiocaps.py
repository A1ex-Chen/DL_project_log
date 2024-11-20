def calculate_selection_performance_clotho_audiocaps(val_metrics_per_dataset):
    """
    Calculate performance for Clotho+AudioCaps for model selection.
    """
    selection_performance_all = []
    for n in val_metrics_per_dataset.keys():
        selection_performance = (val_metrics_per_dataset[n][
            f'{n}/audio_to_text_mAP@10'] + val_metrics_per_dataset[n][
            f'{n}/text_to_audio_mAP@10']) / 2
        selection_performance_all.append(selection_performance)
    return np.mean(selection_performance_all)
