def extract_strings_between_tags(string):
    pattern = '<object>(.*?)</object>'
    matches = re.findall(pattern, string)
    return matches
