def find_pattern_match(filename):
    pattern_list = model_location_patterns()
    return any(re.search(pattern, filename) for pattern in pattern_list)
