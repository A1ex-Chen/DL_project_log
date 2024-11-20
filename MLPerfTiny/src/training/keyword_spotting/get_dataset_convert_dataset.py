def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    audio = item['audio']
    label = item['label']
    return audio, label
