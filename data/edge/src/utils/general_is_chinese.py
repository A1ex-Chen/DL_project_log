def is_chinese(s='人工智能'):
    return bool(re.search('[一-\u9fff]', str(s)))
