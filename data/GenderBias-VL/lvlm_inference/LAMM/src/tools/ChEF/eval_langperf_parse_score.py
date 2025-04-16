def parse_score(text):
    match = re.search('\\[Overall Score\\]\\s*(\\d+(?:\\.\\d+)?)', text)
    if match:
        score = float(match.group(1))
    else:
        print(f'Overall Score not found. origin text: {text}')
        score = None
    return score
