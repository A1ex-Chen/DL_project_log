def find_phrases(text):
    phrases = re.finditer('<phrase>(.*?)</phrase>', text)
    return [(match.group(1), match.start(1), match.end(1)) for match in phrases
        ]
