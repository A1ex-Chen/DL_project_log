def extract_numbers_from_patch(string):
    pattern = '<patch_index_(\\d+)>'
    matches = re.findall(pattern, string)
    return matches
