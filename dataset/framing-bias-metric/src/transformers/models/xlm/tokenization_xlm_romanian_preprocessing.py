def romanian_preprocessing(text):
    """Sennrich's WMT16 scripts for Romanian preprocessing, used by model `xlm-mlm-enro-1024`"""
    text = text.replace('Ş', 'Ș').replace('ş', 'ș')
    text = text.replace('Ţ', 'Ț').replace('ţ', 'ț')
    text = text.replace('Ș', 'S').replace('ș', 's')
    text = text.replace('Ț', 'T').replace('ț', 't')
    text = text.replace('Ă', 'A').replace('ă', 'a')
    text = text.replace('Â', 'A').replace('â', 'a')
    text = text.replace('Î', 'I').replace('î', 'i')
    return text
