def parse_model_name(answer_path):
    js_name = os.path.basename(answer_path)
    match = re.match('([^_]+)', js_name)
    return match.group(1)
