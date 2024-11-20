def parse_caption_sentence(text):
    pattern = '(?<=[\'\\"])(.*?)(?=[\'\\"])'
    sentences = re.findall(pattern, text)
    if len(sentences) > 0:
        return '. '.join(sentences)
    return text
