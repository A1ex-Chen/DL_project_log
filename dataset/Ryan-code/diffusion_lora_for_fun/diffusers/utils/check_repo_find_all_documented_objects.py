def find_all_documented_objects():
    """Parse the content of all doc files to detect which classes and functions it documents"""
    documented_obj = []
    for doc_file in Path(PATH_TO_DOC).glob('**/*.rst'):
        with open(doc_file, 'r', encoding='utf-8', newline='\n') as f:
            content = f.read()
        raw_doc_objs = re.findall(
            '(?:autoclass|autofunction):: transformers.(\\S+)\\s+', content)
        documented_obj += [obj.split('.')[-1] for obj in raw_doc_objs]
    for doc_file in Path(PATH_TO_DOC).glob('**/*.md'):
        with open(doc_file, 'r', encoding='utf-8', newline='\n') as f:
            content = f.read()
        raw_doc_objs = re.findall('\\[\\[autodoc\\]\\]\\s+(\\S+)\\s+', content)
        documented_obj += [obj.split('.')[-1] for obj in raw_doc_objs]
    return documented_obj
