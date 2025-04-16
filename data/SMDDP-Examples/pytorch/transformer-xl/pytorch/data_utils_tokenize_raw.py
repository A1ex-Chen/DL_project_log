def tokenize_raw(text, lang='en'):
    mt = sacremoses.MosesTokenizer(lang)
    text = mt.tokenize(text, return_str=True)
    text = re.sub('&quot;', '"', text)
    text = re.sub('&apos;', "'", text)
    text = re.sub('(\\d)\\.(\\d)', '\\1 @.@ \\2', text)
    text = re.sub('(\\d),(\\d)', '\\1 @,@ \\2', text)
    text = re.sub('(\\w)-(\\w)', '\\1 @-@ \\2', text)
    return text
